"""DDL only — no resampling"""
import pymysql
conn = pymysql.connect(host='localhost',port=3305,user='root',password='',database='forex_predictor',charset='utf8mb4')
cur = conn.cursor()

cur.execute('SET FOREIGN_KEY_CHECKS=0')
for t in ['model_predictions','shap_contributions','model_runs','feature_versions','features_15m']:
    cur.execute(f'DROP TABLE IF EXISTS {t}')
    print(f'DROP {t} OK')
cur.execute('SET FOREIGN_KEY_CHECKS=1')
conn.commit()

cur.execute("""CREATE TABLE feature_versions (
    version_key VARCHAR(50) NOT NULL PRIMARY KEY,
    dimension INT NOT NULL,
    feature_names LONGTEXT,
    feature_count INT,
    status ENUM('active','deprecated','experimental') NOT NULL DEFAULT 'active',
    description TEXT,
    mlflow_run_id VARCHAR(50),
    build_module VARCHAR(255),
    first_trained_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_status(status), INDEX idx_dimension(dimension), INDEX idx_created(created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
conn.commit(); print('CREATE feature_versions OK')

cur.execute("""CREATE TABLE model_runs (
    run_id VARCHAR(50) NOT NULL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    target_tf VARCHAR(5) NOT NULL,
    dimension INT,
    feature_version_key VARCHAR(50),
    auc FLOAT,
    params_json TEXT,
    trials_completed INT DEFAULT 0,
    live_count INT DEFAULT 0,
    live_accuracy FLOAT,
    live_status ENUM('pending','win','lose') DEFAULT 'pending',
    selected_features LONGTEXT COMMENT 'JSON array of feature names used in this run (null=all)',
    description TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_dim(symbol,dimension),
    INDEX idx_target_tf(target_tf),
    INDEX idx_feature_version(feature_version_key),
    INDEX idx_auc(auc),
    CONSTRAINT fk_run_fv FOREIGN KEY(feature_version_key)
        REFERENCES feature_versions(version_key) ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
conn.commit(); print('CREATE model_runs OK')

cur.execute("""CREATE TABLE model_predictions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    target_tf VARCHAR(5) NOT NULL,
    timestamp DATETIME NOT NULL,
    mode ENUM('training','natural') NOT NULL,
    prob_up FLOAT,
    `signal` TINYINT,
    is_adopted TINYINT DEFAULT 0,
    adopted_at DATETIME,
    true_label TINYINT,
    is_correct TINYINT,
    dimension INT,
    INDEX idx_run_ts_mode(run_id,timestamp,mode),
    INDEX idx_tf_symbol_ts(target_tf,symbol,timestamp),
    INDEX idx_symbol_ts(symbol,timestamp),
    INDEX idx_adopted(is_adopted),
    CONSTRAINT fk_pred_run FOREIGN KEY(run_id)
        REFERENCES model_runs(run_id) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
conn.commit(); print('CREATE model_predictions OK')

cur.execute("""CREATE TABLE shap_contributions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    feature_id VARCHAR(20) NOT NULL,
    feature_name VARCHAR(100),
    importance FLOAT,
    rank INT,
    INDEX idx_run(run_id),
    INDEX idx_run_rank(run_id,rank),
    CONSTRAINT fk_shap_run FOREIGN KEY(run_id)
        REFERENCES model_runs(run_id) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
conn.commit(); print('CREATE shap_contributions OK')

cur.execute("""CREATE TABLE features_15m (
    timestamp DATETIME NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    target_15m TINYINT,
    PRIMARY KEY(timestamp,symbol),
    INDEX idx_symbol(symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
conn.commit(); print('CREATE features_15m OK')

cur.close(); conn.close()
print('ALL DDL DONE')
