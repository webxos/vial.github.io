#!/bin/bash
docker-compose down
docker-compose up --build -d
git add . && git commit -m "Deploy update" && git push origin main
