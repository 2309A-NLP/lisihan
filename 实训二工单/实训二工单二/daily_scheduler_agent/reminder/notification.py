# -*- coding: utf-8 -*-
import logging
import subprocess
import threading

logger = logging.getLogger("Notification")
_POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

def show_windows_toast(title, message, duration=8):
    def _show():
        try:
            safe_title = title.replace('"', '`"')
            safe_message = message.replace('"', '`"')
            ps = (
                'Add-Type -AssemblyName System.Windows.Forms\n'
                '$notify = New-Object System.Windows.Forms.NotifyIcon\n'
                '$notify.Icon = [System.Drawing.SystemIcons]::Information\n'
                '$notify.BalloonTipTitle = "' + safe_title + '"\n'
                '$notify.BalloonTipText = "' + safe_message + '"\n'
                '$notify.Visible = $true\n'
                '$notify.ShowBalloonTip(' + str(duration * 1000) + ')\n'
                'Start-Sleep -Seconds ' + str(duration) + '\n'
                '$notify.Dispose()\n'
            )
            subprocess.run(
                [_POWERSHELL, "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                capture_output=True, timeout=duration + 5,
            )
            logger.info("Windows toast sent: %s - %s", title, message)
        except subprocess.TimeoutExpired:
            logger.warning("Windows toast timed out (duration=%ss)", duration)
        except Exception as exc:
            logger.warning("Windows toast failed: %s", exc)
    threading.Thread(target=_show, daemon=True).start()

def notify_reminder(schedule_id, content, scheduled_time):
    show_windows_toast("日程提醒", content + "\n时间：" + scheduled_time)
