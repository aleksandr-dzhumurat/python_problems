# Базовые алгоритмы

## Окружение для разработки в Docker

Готово для использования в vscode (см. `.devcontainer.json`)

```
docker build -t python_problems:latest .
```


## Окружение для разработки через venv

```shell
sudo apt-get install python3-venv && python3 -m venv .env
```

Активация окружения
```shell
source .env/bin/activate
```

Установка зависимостей
```shell
pip3 install -r requirements.txt
```
