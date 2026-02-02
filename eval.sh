# python evaluation.py --model hf --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,dtype=float16 \
#  --tasks mmlu --num_fewshot 5 --device cuda --batch_size 16 --output_path results/llama-8b-alpaca-cleaned.json

# accelerate launch --num_processes 1 -m lm_eval --model hf --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,dtype=float16 \
#  --tasks mmlu --num_fewshot 5 --device cuda --batch_size 16 --output_path results/mmlu_qwen3-4b-instruct-2507.json

# python evaluation.py --model hf --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,dtype=float16 \
#  --tasks gsm8k --num_fewshot 5 --device cuda --batch_size 16 --output_path results/llama-8b-alpaca-cleaned-gsm8k.json

python evaluation.py --model hf --model_args pretrained=checkpoints/llama-8b-alpaca-cleaned,dtype=float16 \
 --tasks hellaswag,arc_easy,arc_challenge --num_fewshot 0 --device cuda --batch_size 16 --output_path results/llama-8b-combined.json