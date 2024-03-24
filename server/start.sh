#!/bin/sh

redis-server &
flask --app server run --host=0.0.0.0