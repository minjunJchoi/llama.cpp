#!/bin/bash

#
# Temporary script - will be removed in the future
#

cd `dirname $0`
cd ..

# Important:
#
#   "--keep 48" is based on the contents of prompts/chat-with-bob.txt
#
#./main -m ./models/llama-7b/ggml-model-q4_0.gguf -c 512 -b 1024 -n 256 --keep 48 \
#    --repeat_penalty 1.0 --color -i \
#    -r "User:" -f prompts/chat-with-bob.txt


./main -ngl 76 -m ./models/llama-2-70b.Q5_K_M.gguf -c 512 -b 1024 -n 256 --keep 48 \
    --repeat_penalty 1.0 --color -i \
    -r "User:" -f prompts/chat-with-bob.txt
