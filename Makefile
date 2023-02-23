.PHONY: up
up:
	docker-compose up --build --force-recreate --remove-orphans

.PHONY: d
d:
	docker-compose up -d --build --force-recreate --remove-orphans

.PHONY: down
down:
	docker-compose down
