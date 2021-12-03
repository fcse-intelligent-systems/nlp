from eval_metrics.tokenizer.ptbtokenizer import PTBTokenizer
from eval_metrics.rouge.rouge import Rouge
from eval_metrics.bleu.bleu import Bleu


def score_predictions(predictions):
    tokenizer = PTBTokenizer()

    gens = {}
    refs = {}

    for p, i in zip(predictions, range(len(predictions))):
        gens[str(i)] = [p['predicted']]
        refs[str(i)] = [p['gt']]

    gens = tokenizer.tokenize(gens)
    refs = tokenizer.tokenize(refs)

    results = dict()

    print('Calculating ROUGE...')
    rouge_scorer = Rouge()
    rouge_avg_score, rouge_all_scores = rouge_scorer.compute_score(refs, gens)
    results['ROUGE'] = (rouge_avg_score, rouge_all_scores)

    print('Calculating BLEU...')
    bleu_scorer = Bleu(4)
    bleu_avg_score, bleu_all_scores = bleu_scorer.compute_score(refs, gens)
    for n, bleu_n_avg_score, bleu_n_all_scores in zip(range(len(bleu_avg_score)), bleu_avg_score, bleu_all_scores):
        results[f'BLEU-{n + 1}'] = (bleu_n_avg_score, bleu_n_all_scores)

    return results
