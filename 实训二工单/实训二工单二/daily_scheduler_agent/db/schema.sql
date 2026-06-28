-- @工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
-- @作者: [AI生成]
-- @功能: MySQL 数据库结构，包含日程表、Agent 执行日志表和提醒日志表

CREATE DATABASE IF NOT EXISTS schedule_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE schedule_db;

CREATE TABLE IF NOT EXISTS schedules (
    id INT NOT NULL AUTO_INCREMENT,
    content VARCHAR(255) NOT NULL COMMENT '日程内容',
    scheduled_time DATETIME NOT NULL COMMENT '首次提醒时间',
    repeat_rule VARCHAR(50) DEFAULT NULL COMMENT '循环规则: daily/weekly/monthly/NULL',
    repeat_end_date DATE DEFAULT NULL COMMENT '循环结束日期',
    status TINYINT NOT NULL DEFAULT 1 COMMENT '1有效 0已删除',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_schedules_time (scheduled_time),
    INDEX idx_schedules_status (status),
    INDEX idx_schedules_repeat (repeat_rule)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='日程表';

CREATE TABLE IF NOT EXISTS agent_execution_logs (
    id INT NOT NULL AUTO_INCREMENT,
    user_input VARCHAR(500) DEFAULT NULL COMMENT '用户原始输入',
    intent VARCHAR(50) DEFAULT NULL COMMENT '识别出的意图',
    action VARCHAR(50) DEFAULT NULL COMMENT 'add/delete/query/update/remind/system',
    target_schedule_id INT DEFAULT NULL COMMENT '操作的日程ID',
    result VARCHAR(50) NOT NULL COMMENT 'success/failed',
    error_message TEXT DEFAULT NULL COMMENT '错误信息',
    execution_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_execution_time (execution_time),
    INDEX idx_execution_action (action),
    INDEX idx_execution_target (target_schedule_id),
    CONSTRAINT fk_execution_logs_schedule
        FOREIGN KEY (target_schedule_id) REFERENCES schedules(id)
        ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Agent任务执行日志表';

CREATE TABLE IF NOT EXISTS reminder_logs (
    id INT NOT NULL AUTO_INCREMENT,
    schedule_id INT NOT NULL,
    occurrence_time DATETIME NOT NULL COMMENT '本次应提醒的发生时间',
    reminder_content VARCHAR(255) DEFAULT NULL COMMENT '提醒内容',
    reminded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_schedule_occurrence (schedule_id, occurrence_time),
    INDEX idx_reminder_logs_schedule (schedule_id),
    INDEX idx_reminded_at (reminded_at),
    CONSTRAINT fk_reminder_logs_schedule
        FOREIGN KEY (schedule_id) REFERENCES schedules(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='提醒发送记录表';
