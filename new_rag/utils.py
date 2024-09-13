from collections import defaultdict

def print_color(text, color):
    """Print text with color in terminal"""
    if color == "red":
        print(f"\033[91m{text}\033[00m")
    elif color == "green":
        print(f"\033[92m{text}\033[00m")
    elif color == "yellow":
        print(f"\033[93m{text}\033[00m")
    elif color == "blue":
        print(f"\033[94m{text}\033[00m")
    elif color == "purple":
        print(f"\033[95m{text}\033[00m")
    elif color == "cyan":
        print(f"\033[96m{text}\033[00m")
    elif color == "white":
        print(f"\033[97m{text}\033[00m")
    else:
        print(f"\033[00m{text}\033[00m")

def rabin_karp(text, M, q):
    if M == 0: return True
    h, t, d = (1<<(8*M-8))%q, 0, 256

    dic = defaultdict(list)

    for i in range(M):
        t = (d * t + ord(text[i]))% q

    dic[t].append(i-M+1)

    for i in range(len(text) - M):
        t = (d*(t-ord(text[i])*h) + ord(text[i + M]))% q
        for j in dic[t]:
            if text[i+1:i+M+1] == text[j:j+M]:
                return True, text[j:j + M]
        dic[t].append(i+1)
    return False, ""

def longest_dup_substring(s):
    beg, end = 0, len(s)
    q = (1<<31) - 1
    found = ""
    while beg + 1 < end:
        mid = (beg + end)//2
        is_found, candidate = rabin_karp(s, mid, q)
        if is_found:
            beg, found = mid, candidate
        else:
            end = mid

    return found

def post_process_answer(answer):
    # Remove duplication
    dup = longest_dup_substring(answer)
    new_answer = answer.replace(dup, "")

    # Sometime, the model just repeat the question
    new_answer = new_answer.split("Question:")[0]
    return new_answer
