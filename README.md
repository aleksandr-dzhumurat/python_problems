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
python -m pip install -r requirements.txt
```

## Окружение для Mac

```shell
brew install openssl xz gdbm
```

```shell
brew install pyenv-virtualenv
```

```shell
pyenv install 3.12
```

```shell
pyenv virtualenv 3.12 problems-env
```


```shell
    source ~/.pyenv/versions/problems-env/bin/activate
```


