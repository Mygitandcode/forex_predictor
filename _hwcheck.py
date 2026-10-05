"""Quick hardware info"""
import subprocess, json

def pwsh(cmd):
    return subprocess.check_output(['powershell', '-Command', cmd]).decode('utf-8', errors='ignore')

# CPU
cpu = pwsh("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json")
cpu = json.loads(cpu)
if isinstance(cpu, dict): cpu = [cpu]
c = cpu[0]
print(f"CPU: {c['Name']}")
print(f"  核心: {c['NumberOfCores']}核 {c['NumberOfLogicalProcessors']}线程, 基频 {c['MaxClockSpeed']}MHz ({c['MaxClockSpeed']//1000}.{c['MaxClockSpeed']%1000//100}GHz)")

# GPU
gpu = pwsh("Get-CimInstance Win32_VideoController | Where-Object { $_.Status -eq 'OK' } | Select-Object Name,AdapterRAM | ConvertTo-Json")
gpu = json.loads(gpu)
if isinstance(gpu, dict): gpu = [gpu]
for g in gpu:
    vram = g['AdapterRAM'] / (1024**3) if g['AdapterRAM'] else 0
    name = g['Name'][:80]
    print(f"GPU: {name}")
    print(f"  显存: {vram:.1f} GB" if vram > 0 else "  显存: (集显/共享)")

# RAM
mem = pwsh("Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,Manufacturer | ConvertTo-Json")
mem = json.loads(mem)
if isinstance(mem, dict): mem = [mem]
total = sum(m['Capacity'] for m in mem) / (1024**3)
m0 = mem[0]
print(f"RAM: {total:.0f} GB DDR4-{m0['Speed']}MHz ({m0['Manufacturer']}) - {len(mem)}条")

# Disks
disk = pwsh("Get-CimInstance Win32_DiskDrive | Select-Object Model,Size,MediaType | ConvertTo-Json")
disk = json.loads(disk)
if isinstance(disk, dict): disk = [disk]
print("磁盘:")
for d in disk:
    size = d['Size'] / (1024**3) if d['Size'] else 0
    media = d.get('MediaType', 'Unknown')
    print(f"  {d['Model'][:60]} - {size:.0f}GB ({media})")
