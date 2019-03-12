$task="joint"
$model_choice="bi-ltsm-crf"

echo "Running Bi-LSTM with Character Embeddings Monolingual: English"
python ../main.py --task=$task --mode="mono" --model-choice=$model_choice --train-lang="English" --test-lang="English"

echo "Running Bi-LSTM with Character Embeddings Monolingual: Chinese"
python ../main.py --task=$task --mode="mono" --model-choice=$model_choice --train-lang="Chinese" --test-lang="Chinese"

echo "Running Bi-LSTM with Character Embeddings Monolingual: Arabic"
python ../main.py --task=$task --mode="mono" --model-choice=$model_choice --train-lang="Arabic" --test-lang="Arabic"

#################################### MULTILINGUAL EXPERIMENTS USING OFFLINE EMBEDDINGS ################################################

echo "Running Bi-LSTM with Character Multilingual Embeddings training and testing on All Languages: English, Chinese, Arabic"
python ../main.py --task=$task --mode="multi" --model-choice=$model_choice --train-lang="English,Chinese,Arabic" --test-lang="English,Chinese,Arabic" --multi-model-file="multi_expert_dict"

echo "Running Bi-LSTM with Character Multilingual Embeddings training on English only and testing on All Languages: English, Chinese, Arabic"
python ../main.py --task=$task" --mode="multi" --model-choice=$model_choice --train-lang="English" --test-lang="English,Chinese,Arabic" --multi-model-file="multi_expert_dict"

echo "Running Bi-LSTM with Character Multilingual Embeddings training on English and Arabic and testing on All Languages: English, Chinese, Arabic"
python ../main.py --task=$task --mode="multi" --model-choice=$model_choice --train-lang="English,Arabic" --test-lang="English,Chinese,Arabic" --multi-model-file="multi_expert_dict"

echo "Running Bi-LSTM with Character Multilingual Embeddings training on English and Chinese and testing on All Languages: English, Chinese, Arabic"
python ../main.py --task=$task --mode="multi" --model-choice=$model_choice --train-lang="English,Chinese" --test-lang="English,Chinese,Arabic" --multi-model-file="multi_expert_dict"

#################################### MULTILINGUAL EXPERIMENTS USING BERT ################################################

echo "Running Bi-LSTM with Character Multilingual Embeddings training and testing on All Languages: English, Chinese, Arabic"
python ../main.py --task=$task --mode="multi" --model-choice=$model_choice --train-lang="English,Chinese,Arabic" --test-lang="English,Chinese,Arabic" --multi-model-file="bert"

echo "Running Bi-LSTM with Character Multilingual Embeddings training on English only and testing on All Languages: English, Chinese, Arabic"
python ../main.py --task=$task --mode="multi" --model-choice=$model_choice --train-lang="English" --test-lang="English,Chinese,Arabic" --multi-model-file="bert"

echo "Running Bi-LSTM with Character Multilingual Embeddings training on English and Arabic and testing on All Languages: English, Chinese, Arabic"
python ../main.py --task=$task --mode="multi" --model-choice=$model_choice --train-lang="English,Arabic" --test-lang="English,Chinese,Arabic" --multi-model-file="bert"

echo "Running Bi-LSTM with Character Multilingual Embeddings training on English and Chinese and testing on All Languages: English, Chinese, Arabic"
python ../main.py --task=$task --mode="multi" --model-choice=$model_choice --train-lang="English,Chinese" --test-lang="English,Chinese,Arabic" --multi-model-file="bert"
