#!/bin/bash

curl -X POST localhost:8001/score -H 'Content-Type: application/json' \
      -d '{
      "messages": [
              {"subject": "URGENT: You have WON!!!", "body": "Click here now to claim your $1000000 prize!!! Limited time offer!!!"},
              {"subject": "Q3 planning notes", "body": "Attached is the agenda for tomorrow'\''s meeting. Let me know if I missed anything."}
      ]
}'
