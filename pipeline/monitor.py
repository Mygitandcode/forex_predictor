"""
pipeline/monitor.py — 桌面悬浮资源监控 + MQ 队列状态

显示: 内存/CPU Top-8 进程 + Redis MQ 队列状态。
可最小化、关闭，始终置顶。每 2 秒刷新。
"""

import sys, os
# 确保项目根在 sys.path (Start-Process 可能改变 CWD)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import tkinter as tk
import psutil

# ---- 配置 ----
FONT = ("Consolas", 9)
FONT_BOLD = ("Consolas", 9, "bold")
REFRESH_SEC = 2
MAX_PROCS = 8
ROW_HEIGHT = 22


class MonitorWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("资源监控 + MQ 队列")
        self.root.attributes("-topmost", True)

        screen_w = self.root.winfo_screenwidth()
        win_h = 90 + MAX_PROCS * ROW_HEIGHT
        self.root.geometry(f"440x{win_h}+{screen_w // 8}+20")
        self.root.minsize(380, win_h)
        self.root.configure(bg="#1a1a2e")

        # ---- 标题行 ----
        self.title_label = tk.Label(
            self.root, text=" 资源监控 (2s)  PID | 内存(MB) | CPU% | 进程名",
            font=FONT_BOLD, fg="#e94560", bg="#1a1a2e", anchor="w", height=1
        )
        self.title_label.pack(fill="x", pady=(4, 2))

        sep = tk.Frame(self.root, height=1, bg="#333355")
        sep.pack(fill="x")

        # ---- 进程列表 ----
        self.lines = []
        for i in range(MAX_PROCS):
            lbl = tk.Label(
                self.root, text="", font=FONT,
                fg="#16c79a", bg="#1a1a2e", anchor="w", height=1
            )
            lbl.pack(fill="x", padx=6, pady=1)
            self.lines.append(lbl)

        # ---- 分隔 + MQ 状态 ----
        sep2 = tk.Frame(self.root, height=1, bg="#333355")
        sep2.pack(fill="x")

        self.mq_label = tk.Label(
            self.root, text="", font=FONT,
            fg="#f9ed69", bg="#1a1a2e", anchor="w", height=1
        )
        self.mq_label.pack(fill="x", padx=6, pady=1)

        self.stage_label = tk.Label(
            self.root, text="", font=FONT,
            fg="#a2d2ff", bg="#1a1a2e", anchor="w", height=1
        )
        self.stage_label.pack(fill="x", padx=6, pady=1)

        # ---- 底栏 ----
        sep3 = tk.Frame(self.root, height=1, bg="#333355")
        sep3.pack(fill="x")

        self.footer = tk.Label(
            self.root, text="", font=FONT_BOLD,
            fg="#e94560", bg="#16213e", anchor="w", height=1
        )
        self.footer.pack(fill="x", pady=(2, 4), ipady=2)

        # Redis 连接 (延迟初始化, 避免阻塞启动)
        self._redis = None
        self._try_redis()

        self.running = True
        self._refresh()

    def _try_redis(self):
        try:
            # 使用与 pipeline/mq.py 完全相同的连接 (已验证可用)
            from pipeline.mq import get_redis
            self._redis = get_redis()
            self._redis.ping()
        except Exception as e:
            self._redis = None
            if not getattr(self, '_mq_err_logged', False):
                print(f"[monitor] Redis connect fail: {e}", flush=True)
                self._mq_err_logged = True

    def _refresh(self):
        if not self.running:
            return

        try:
            # ---- 进程 Top-8 ----
            procs = []
            for p in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
                try:
                    info = p.info
                    mb = info['memory_info'].rss / (1024 * 1024)
                    procs.append((info['pid'], mb, info['cpu_percent'] or 0, info['name'] or '?'))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            procs.sort(key=lambda x: x[1], reverse=True)
            top = procs[:MAX_PROCS]

            colors = ["#16c79a", "#a2d2ff", "#ffc93c", "#f9ed69",
                      "#f08a5d", "#b83b5e", "#6a2c70", "#00bcd4"]
            for i, (pid, mb, cpu, name) in enumerate(top):
                self.lines[i].config(
                    text=f"  {pid:>6d}  {mb:>10.1f}  {cpu:>4.1f}  {name[:30]}",
                    fg=colors[i % len(colors)]
                )
            for i in range(len(top), MAX_PROCS):
                self.lines[i].config(text="")

            # ---- MQ 队列状态 ----
            if self._redis:
                try:
                    qlen = self._redis.llen("training:requests")
                    self.mq_label.config(
                        text=f"  MQ(Redis): {qlen} 条待处理" if qlen else "  MQ(Redis): 空闲 (等待消息...)",
                        fg="#ff6b6b" if qlen else "#51cf66"
                    )

                    # 查找活跃 pipeline run（优先 running > 最近更新的 > 第一个）
                    keys = self._redis.keys("pipeline:run:*")
                    if keys:
                        best_key = None
                        best_info = None
                        for k in keys:
                            info = self._redis.hgetall(k)
                            status = info.get("status", "")
                            # 优先 running
                            if status == "running":
                                best_key, best_info = k, info
                                break
                            if best_key is None:
                                best_key, best_info = k, info
                        if best_key and best_info:
                            name = (best_key.decode() if isinstance(best_key, bytes) else best_key).replace("pipeline:run:", "")
                            s = [best_info.get(f"stage-{i}", "-") for i in range(1, 7)]
                            self.stage_label.config(
                                text=f"  Run: {name[:22]} | S1={s[0]} S2={s[1]} S3={s[2]} S4={s[3]} S5={s[4]} S6={s[5]}",
                                fg="#a2d2ff"
                            )
                    else:
                        self.stage_label.config(text="  Run: 无活跃任务", fg="#666688")
                except Exception as e:
                    self.mq_label.config(text=f"  MQ(Redis): 错误", fg="#ff6b6b")
                    if not getattr(self, '_mq_err2_logged', False):
                        print(f"[monitor] MQ query fail: {e}", flush=True)
                        self._mq_err2_logged = True
            else:
                self.mq_label.config(text="  MQ(Redis): 未连接", fg="#ff6b6b")
                self._try_redis()

            # ---- 底栏 ----
            mem_total = psutil.virtual_memory()
            cpu_total = psutil.cpu_percent(interval=0.1)
            self.footer.config(
                text=f"  总内存: {mem_total.used/(1024**3):.1f}/{mem_total.total/(1024**3):.1f}GB "
                     f"({mem_total.percent}%) | CPU: {cpu_total:.0f}%"
            )

        except Exception:
            import traceback; traceback.print_exc()
            pass

        self.root.after(REFRESH_SEC * 1000, self._refresh)

    def run(self):
        self.root.mainloop()
        self.running = False


if __name__ == "__main__":
    MonitorWindow().run()
