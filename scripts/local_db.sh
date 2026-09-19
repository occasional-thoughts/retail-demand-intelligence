#!/usr/bin/env bash
# Start PostgreSQL and MongoDB as ordinary user processes, no Docker.
#
# docker-compose.yml is the primary, reproducible path and is what the README
# documents first. This script exists because Docker Desktop on macOS needs an
# administrator password and a GUI dialog, which is a poor dependency for a
# pipeline that otherwise runs headless -- and an impossible one on a managed
# or locked-down machine.
#
# Both paths produce the same two databases on the same two ports with the same
# credentials, so everything downstream is identical either way.
#
# Usage:  ./scripts/local_db.sh start | stop | status
set -euo pipefail

CONDA_PREFIX_DB="${CONDA_PREFIX_DB:-/opt/anaconda3/envs/retaildb}"
DATA_ROOT="${DATA_ROOT:-$HOME/.retail-intel-db}"
PGDATA="$DATA_ROOT/pgdata"
MONGODATA="$DATA_ROOT/mongodata"
LOGS="$DATA_ROOT/logs"

PG_USER=retail
PG_PASSWORD=retail
PG_DB=retail
PG_PORT=5433
MONGO_PORT=27017

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/retail/bin/python}"
export PGPASSWORD="retail"        # non-interactive: psql must never prompt
export PATH="$CONDA_PREFIX_DB/bin:$PATH"

mkdir -p "$LOGS" "$MONGODATA"

start_postgres() {
    if [ ! -d "$PGDATA/base" ]; then
        echo "initialising postgres cluster at $PGDATA"
        mkdir -p "$PGDATA"
        echo "$PG_PASSWORD" > "$DATA_ROOT/.pgpass_init"
        initdb -D "$PGDATA" -U "$PG_USER" --auth=md5 \
               --pwfile="$DATA_ROOT/.pgpass_init" >/dev/null
        rm -f "$DATA_ROOT/.pgpass_init"
    fi

    if pg_isready -h localhost -p "$PG_PORT" -q 2>/dev/null; then
        echo "postgres already running on :$PG_PORT"
    else
        pg_ctl -D "$PGDATA" -l "$LOGS/postgres.log" \
               -o "-p $PG_PORT -k $DATA_ROOT" start
        until pg_isready -h localhost -p "$PG_PORT" -q 2>/dev/null; do sleep 1; done
        echo "postgres up on :$PG_PORT"
    fi

    # the application database, created once
    if ! psql -h localhost -p "$PG_PORT" -U "$PG_USER" -lqt 2>/dev/null \
         | cut -d'|' -f1 | grep -qw "$PG_DB"; then
        # UTF8 explicitly: the cluster is initdb'd under a C locale, whose
        # default is SQL_ASCII. That would store the Portuguese text in this
        # dataset as unvalidated bytes -- accepted on write, wrong on read.
        createdb -h localhost -p "$PG_PORT" -U "$PG_USER" \
                 -E UTF8 --lc-collate=C --lc-ctype=C -T template0 "$PG_DB"
        echo "created database '$PG_DB'"
    fi
}

start_mongo() {
    if nc -z localhost "$MONGO_PORT" 2>/dev/null; then
        echo "mongodb already running on :$MONGO_PORT"
        return
    fi

    # The conda-forge mongodb package ships mongod but not mongosh, so the
    # bootstrap user is created with pymongo instead. This runs once: mongod
    # is started without auth, the root user is created, mongod is restarted
    # with auth on. The result is a server that takes exactly the same
    # connection URI as the docker-compose one, so nothing downstream has to
    # know which path was used.
    if [ ! -f "$DATA_ROOT/.mongo_user_created" ]; then
        echo "bootstrapping mongodb user (first run)"
        mongod --dbpath "$MONGODATA" --port "$MONGO_PORT" \
               --logpath "$LOGS/mongod-init.log" --fork >/dev/null
        until nc -z localhost "$MONGO_PORT" 2>/dev/null; do sleep 1; done

        "$PYTHON_BIN" - <<'PY'
from pymongo import MongoClient
from pymongo.errors import OperationFailure

client = MongoClient("mongodb://localhost:27017/")
try:
    client.admin.command("createUser", "retail", pwd="retail",
                         roles=[{"role": "root", "db": "admin"}])
    print("created mongo root user 'retail'")
except OperationFailure as error:
    if "already exists" in str(error):
        print("mongo user already exists")
    else:
        raise
client.close()
PY

        mongod --dbpath "$MONGODATA" --shutdown >/dev/null 2>&1 || true
        sleep 2
        touch "$DATA_ROOT/.mongo_user_created"
    fi

    mongod --dbpath "$MONGODATA" --port "$MONGO_PORT" --auth \
           --logpath "$LOGS/mongod.log" --fork >/dev/null
    until nc -z localhost "$MONGO_PORT" 2>/dev/null; do sleep 1; done
    echo "mongodb up on :$MONGO_PORT (auth enabled)"
}

case "${1:-start}" in
    start)
        start_postgres
        start_mongo
        echo ""
        echo "PG_URI    postgresql://retail:retail@localhost:$PG_PORT/retail"
        echo "MONGO_URI mongodb://retail:retail@localhost:27017/?authSource=admin"
        ;;
    stop)
        pg_ctl -D "$PGDATA" stop 2>/dev/null || echo "postgres not running"
        mongod --dbpath "$MONGODATA" --shutdown 2>/dev/null \
            || echo "mongodb not running"
        ;;
    status)
        pg_isready -h localhost -p "$PG_PORT" || true
        nc -z localhost "$MONGO_PORT" && echo "mongodb: accepting connections" \
                                      || echo "mongodb: down"
        ;;
    *)
        echo "usage: $0 {start|stop|status}" >&2
        exit 1
        ;;
esac
