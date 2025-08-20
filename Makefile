.PHONY: build run test clean

build:
    docker-compose build

run:
    docker-compose up

test:
    pytest tests/ -v

clean:
    docker-compose down -v
    rm -rf __pycache__ *.log data/*.db
