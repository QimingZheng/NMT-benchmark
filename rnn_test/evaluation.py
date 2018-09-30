import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

answer_txt = []
with open("answer.txt", 'r') as file:
    answer_txt = file.readlines()
    for i in range(len(answer_txt)):
        answer_txt[i] = answer_txt[i].split()
prediction_txt = []
with open("prediction.txt", 'r') as file:
    prediction_txt = file.readlines()
    for i in range(len(prediction_txt)):
        prediction_txt[i] = prediction_txt[i].split()
        while (prediction_txt[i][-1] == '2'):
            prediction_txt[i].pop()
#            print(len(prediction_txt[i]),)

total_score = 0.0

for i in range(len(prediction_txt)):
    total_score += sentence_bleu([answer_txt[i]], prediction_txt[i])
    #total_score += corpus_bleu([answer_txt[i]], [prediction_txt[i]])

print("BLEU: ", (1.0 * total_score) / len(prediction_txt))
