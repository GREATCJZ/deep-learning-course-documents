import os
import sys
import numpy as np
import nltk
import distance
from http import HTTPStatus
from albumentations.pytorch.transforms import ToTensorV2
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from image_to_latex.lit_models import LitResNetTransformer
global lit_model
global transform
import threading
# python scripts/download_checkpoint.py
def score_files(formulas_ref, formulas_hyp):
    """Loads result from file and score it

    Args:
        path_ref: (string) formulas of reference
        path_hyp: (string) formulas of prediction.

    Returns:
        scores: (dict)

    """
    # load formulas
    # formulas_ref = load_formulas(path_ref)
    # formulas_hyp = load_formulas(path_hyp)

    assert len(formulas_ref) == len(formulas_hyp)

    # tokenize
    refs = [ref.split(" ") for ref in formulas_ref]
    hyps = [hyp.split(" ") for  hyp in formulas_hyp]

    # score
    return {
        "BLEU-4": bleu_score(refs, hyps)*100,
        "ExactMatchScore": exact_match_score(refs, hyps)*100,
        "EditDistance": edit_distance(refs, hyps)*100
    }


def exact_match_score(references, hypotheses):
    """Computes exact match scores.

    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)

    Returns:
        exact_match: (float) 1 is perfect

    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """Computes bleu score.

    Args:
        references: list of list (one hypothesis)
        hypotheses: list of list (one hypothesis)

    Returns:
        BLEU-4 score: (float)

    """
    references = [[ref] for ref in references]  # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_4


def edit_distance(references, hypotheses):
    """Computes Levenshtein distance between two sequences.

    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)

    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)

    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot


def truncate_end(list_of_ids, id_end):
    """Removes the end of the list starting from the first id_end token"""
    list_trunc = []
    for idx in list_of_ids:
        if idx == id_end:
            break
        else:
            list_trunc.append(idx)

    return list_trunc


def write_answers(references, hypotheses, rev_vocab, dir_name, id_end):
    """Writes text answers in files.

    One file for the reference, one file for each hypotheses

    Args:
        references: list of list         (one reference)
        hypotheses: list of list of list (multiple hypotheses)
            hypotheses[0] is a list of all the first hypothesis for all the
            dataset
        rev_vocab: (dict) rev_vocab[idx] = word
        dir_name: (string) path where to write results
        id_end: (int) special id of token that corresponds to the END of
            sentence

    Returns:
        file_names: list of the created files

    """
    def ids_to_str(ids):
        ids = truncate_end(ids, id_end)
        s = [rev_vocab[idx] for idx in ids]
        return " ".join(s)

    def write_file(file_name, list_of_list):
        print("writting file ", file_name)
        with open(file_name, "w") as f:
            for l in list_of_list:
                f.write(ids_to_str(l) + "\n")

    init_dir(dir_name)
    file_names = [dir_name + "ref.txt"]
    write_file(dir_name + "ref.txt", references)  # one file for the ref
    for i in range(len(hypotheses)):              # one file per hypo
        assert len(references) == len(hypotheses[i])
        write_file(dir_name + "hyp_{}.txt".format(i), hypotheses[i])
        file_names.append(dir_name + "hyp_{}.txt".format(i))

    return file_names


lit_model = LitResNetTransformer.load_from_checkpoint("./artifacts/model.pt")
lit_model.freeze()
transform = ToTensorV2()
PATH="/root/autodl-tmp/image-to-latex-main/data/formula_images_processed/"
core=8



def fun1(pid):
    global lit_model
    pid=eval(pid)
    formula_list=[]
    formula_pred_list=[]
    with open("./data/im2latex_formulas.norm.new.lst",'r') as F:
        Formula=F.readlines()
        with open("./data/im2latex_validate_filter.lst",'r') as f:
            txt=f.readlines()
            length=len(txt)
            inter_l=int(length/(core-1))*pid
            inter_r=min(length, int(length/(core-1))*(pid+1))
            for i in range(inter_l,inter_r):
                line=txt[i]
                line=line.replace("\n",'')
                image_name=line.split(" ")[0]
                formula_idx=eval(line.split(" ")[1])
                formula=Formula[formula_idx].replace("\n",'')
                path = PATH + image_name
                if line == "65147.png" or line == "78755.png" or line == "84261.png" or line == "78755.png":
                    formula_pred=" "
                else:
                    image = Image.open(path).convert("L")
                    image_tensor = transform(image=np.array(image))["image"]
                    pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())[0].numpy()
                    decoded = lit_model.tokenizer.decode(list(pred))
                    formula_pred = " ".join(decoded)
                    print(formula_pred)
                    formula_list.append(formula)
                    formula_pred_list.append(formula_pred)
    with open("./scripts/devtest"+str(pid)+".txt", 'w') as f:
        for i in range(0,len(formula_list)):
            f.write(formula_list[i]+"@"+formula_pred_list[i]+"\n")
t0 = threading.Thread(target=fun1, args=("0",))
t1 = threading.Thread(target=fun1, args=("1",))
t2 = threading.Thread(target=fun1, args=("2",))
t3 = threading.Thread(target=fun1, args=("3",))
t4 = threading.Thread(target=fun1, args=("4",))
t5 = threading.Thread(target=fun1, args=("5",))
t6 = threading.Thread(target=fun1, args=("6",))
t7 = threading.Thread(target=fun1, args=("7",))
t0.setDaemon(True)   #把子进程设置为守护线程，必须在start()之前设置
t1.setDaemon(True)
t2.setDaemon(True)
t3.setDaemon(True)
t4.setDaemon(True)
t5.setDaemon(True)
t6.setDaemon(True)
t7.setDaemon(True)
t0.start()
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t0.join()
t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()

formula_tlist=[]
formula_pred_tlist=[]
for i in range(0,core):
    with open("./scripts/devtest"+str(i)+".txt",'r') as f:
        txt=f.readlines()
        for line in txt:
              line=line.replace("\n",'')
              formula_tlist.append(line.split("@")[0])
              formula_pred_tlist.append(line.split("@")[1])
dic=score_files(formula_tlist, formula_pred_tlist)
print(dic)



