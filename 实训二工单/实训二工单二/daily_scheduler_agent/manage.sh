#!/usr/bin/env bash
# @工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
# @作者: [AI生成]
# @功能: 日程提醒智能体后台进程管理脚本

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/agent.log"
PID_FILE="$LOG_DIR/scheduler_agent.pid"
DAEMON_FILE="$SCRIPT_DIR/scheduler_daemon.py"
INTERACTIVE_FILE="$SCRIPT_DIR/interactive.py"

mkdir -p "$LOG_DIR"

is_running() {
    if [[ ! -f "$PID_FILE" ]]; then
        return 1
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start() {
    if is_running; then
        echo "Scheduler Agent is already running with PID $(cat "$PID_FILE")"
        return 0
    fi

    nohup "$PYTHON_BIN" "$DAEMON_FILE" >> "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 1

    if kill -0 "$pid" 2>/dev/null; then
        echo "Scheduler Agent started with PID $pid"
    else
        echo "Scheduler Agent failed to start. Check logs: $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop() {
    if ! is_running; then
        echo "Scheduler Agent is not running"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid"

    for _ in {1..20}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$PID_FILE"
            echo "Scheduler Agent stopped"
            return 0
        fi
        sleep 0.5
    done

    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "Scheduler Agent force stopped"
}

status() {
    if is_running; then
        echo "Scheduler Agent is running with PID $(cat "$PID_FILE")"
    else
        echo "Scheduler Agent is not running"
    fi
}

logs() {
    touch "$LOG_FILE"
    tail -f "$LOG_FILE"
}

restart() {
    stop
    start
}

chat() {
    "$PYTHON_BIN" "$INTERACTIVE_FILE"
}

case "${1:-}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    restart)
        restart
        ;;
    chat)
        chat
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|restart|chat}"
        exit 1
        ;;
esac
