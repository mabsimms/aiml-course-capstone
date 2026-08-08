#!/bin/bash

curl -X POST localhost:8001/score -H 'Content-Type: application/json' \
	-d '{"messages": [{"subject": "URGENT", "body": "click here to claim your prize!!!"}]}'
