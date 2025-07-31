pip install alembic
alembic init alembic

- create a env.py file and use model path to provide for bootstrap
-export DATABASE_URL=postgresql://postgres:pass@localhost:5432/db
-make model changes
-alembic revision --autogenerate -m "Short message about this change"
-alembic upgrade head
-then in docker
-docker compose exec postgres psql -U postgres -d db -c '\dt'
