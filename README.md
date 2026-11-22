## Steps To Run

### Setup

1. Power the ESP32 and connect the PC to MotorMonitor-T15 WiFi (password: motor-t15)

2. In a terminal on the PC, activate the venv (source .venv/bin/activate)

### Logging

To start the logger, the command is in the format:
`python server/MotorLogger.py --label {condition} --duration {duration}`

Note: {duration} is optional and pressing Ctrl+C will also end the logging

For Normal Operation:
`python server/MotorLogger.py --label stable --duration {duration}`

For Overload:
`python server/MotorLogger.py --label overload --duration {duration}`

For Blocked Airflow:
`python server/MotorLogger.py --label "airflow blocked" --duration {duration}`

For Misalignment:
`python server/MotorLogger.py --label misalignment --duration {duration}`

### Training

`python server/Dataset.py`

`python server/Train_RF.py`
