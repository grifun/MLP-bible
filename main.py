import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tokenize import tokenize
import random
from sklearn.neighbors import KernelDensity
import scipy.stats

N_OLD_TESTAMENT = 39

class Book():
    def __init__(self, bookname: str):
        self.name = bookname
        self.sections = []
        self.verses = []
    
    def add_verse(self, section_number: int, verse_number: int, verse_text: list):
        verse = Verse(verse_number, verse_text)
        self.verses.append(verse)
        if self.sections == []:
            self.sections = [ [verse] ]
        elif len(self.sections) == (section_number):
            self.sections[-1].append( verse )
        elif len(self.sections) == (section_number-1):
            self.sections.append( [verse] )
        else:
            raise Exception("wrong section number")
        
    def get_text_length(self):
        if not hasattr(self, "text_length"):
            temp_sum = 0
            for verse in self.verses:
                temp_sum += len(verse.as_array_of_words)
            self.text_length = temp_sum
        return self.text_length

    def get_verses_lengths(self):
        if not hasattr(self, "verses_lengths"):
            self.verses_lengths = [ len(verse.as_array_of_words) for verse in self.verses ]
        return self.verses_lengths
    
    def get_average_verse_lengths(self):
        return np.mean( self.get_verses_lengths() )

def load_stopwords(filename: str):
    stopwords = []
    with open(filename) as file:
        while True:
            line = file.readline()
            if line == "":
                break
            line = line.replace("\n", "")
            stopwords.append(line)
    return stopwords

class Verse():
    IGNORED_WORDS = ["a", "an", "the", "and", "of"]#["a", "an", "and", "in", "of", "or", "that", "the", "to", "unto", "she", "he", "is", "they", "as", "are", "with", "it"]
    STOP_WORDS = load_stopwords("stopwords-big.txt")
    SKIP_CHARS = [",", ".", ";", "?", "!", "'s", "(", ")"]
    def __init__(self, number: int, text: list):
        self.number = number
        self.as_array_of_words = text[:]
        self.as_string = "".join(text)
        tokens = [word.lower() for word in text]
        for skip_char in self.SKIP_CHARS:
            for i in range(len(tokens)):
                tokens[i] = tokens[i].replace(skip_char, "")
        self.as_tokens = [token for token in tokens if token not in self.IGNORED_WORDS]
        self.as_filtered_tokens = [token for token in tokens if token not in self.STOP_WORDS]
        self.as_unique_tokens = np.unique( self.as_tokens )


def parse_line_from_file(line: str) -> (str, int, int, list):
    colon_splitted = line.split(":")
    left_part = colon_splitted[0].split()
    bookname = " ".join( left_part[0:-1] )
    section_number = int( left_part[-1] )
    right_part = " ".join(colon_splitted[1:]).split() 
    verse_number = int( right_part[0] )
    rest = right_part[1:]
    return bookname, section_number, verse_number, rest


def parse_textfile(filename: str) -> dict[Book]:
    books = []
    with open(filename) as file:
        while True:
            line = file.readline()
            if line == "":
                break
            bookname, section_number, verse_number, text = parse_line_from_file(line)
            if books == [] or books[-1].name != bookname:
                books.append( Book(bookname) )

            book = books[-1]
            book.add_verse(section_number, verse_number, text)     
    return books 

def compute_verses_lengths(books: list):
    verses_lengths = {}
    for book in books:
        verses_lengths[book.name] = book.get_verses_lengths()

    average_verses_lengths = {}
    for book in books:
        average_verses_lengths[book.name] = book.get_average_verse_lengths()
    
    return verses_lengths, average_verses_lengths

def add_word_to_dict(word: str, d: dict):
        if word in d:
            d[word] += 1
        else:
            d[word] = 1

def get_wordcounts(books: list, filtered=False):
    OT_counts = {}
    NT_counts = {}
    all_counts = {}

    for book in books[:N_OLD_TESTAMENT]:
        for verse in book.verses:
            if filtered:
                for word in verse.as_filtered_tokens:
                    add_word_to_dict(word, OT_counts)
                    add_word_to_dict(word, all_counts)
            else:
                for word in verse.as_tokens:
                    add_word_to_dict(word, OT_counts)
                    add_word_to_dict(word, all_counts)

    for book in books[N_OLD_TESTAMENT+1:]:
        for verse in book.verses:
            if filtered:
                for word in verse.as_filtered_tokens:
                    add_word_to_dict(word, NT_counts)
                    add_word_to_dict(word, all_counts)
            else:
                for word in verse.as_tokens:
                    add_word_to_dict(word, NT_counts)
                    add_word_to_dict(word, all_counts)
    return OT_counts, NT_counts, all_counts

def get_wordcounts_from_data(data: list, filtered=False):
    OT_counts = {}
    NT_counts = {}
    all_counts = {}
    for verse,y in data:
        if filtered:
            for word in verse.as_filtered_tokens:
                add_word_to_dict(word, all_counts)
                if   y==0:
                    add_word_to_dict(word, OT_counts)
                elif y==1:
                    add_word_to_dict(word, NT_counts)
        else:
            for word in verse.as_tokens:
                add_word_to_dict(word, all_counts)
                if   y==0:
                    add_word_to_dict(word, OT_counts)
                elif y==1:
                    add_word_to_dict(word, NT_counts)
    return OT_counts, NT_counts, all_counts

def get_total_word_count(books):
    total_word_count = 0
    all_words = []
    for book in books:
        total_word_count += book.get_text_length()
        for verse in book.verses:
            all_words += verse.as_array_of_words
    return total_word_count, len(set(all_words))

def get_token_NT_odds(OT_counts, NT_counts, total_word_count):
    odds = {}
    for word in total_word_count.keys():
        if word not in NT_counts:
            odds[word] = 0.
        elif word in NT_counts and word not in OT_counts:
            odds[word] = np.inf
        else:
            odds[word] = NT_counts[word]/OT_counts[word]
    return odds

def odds2logOdds(odds):
    logodds = {}
    for word in odds.keys():
        if odds[word] == 0.:
            logodds[word] = -np.inf
        elif odds[word] == np.inf:
            logodds[word] = np.inf
        else:
            logodds[word] = np.log2(odds[word])
    return logodds

def print_dict(d: dict, limit: int):
    i = 0
    for key, value in d.items():
        print(key, ":", value)
        i+=1
        if i >= limit:
            break

def dicts_into_arrays(OT_dict, NT_dict, all_dict, limit=30):
    words = []
    OT_counts = []
    NT_counts = []
    i = 0
    for word, count in all_dict.items():
        words.append(word)
        OT_counts.append(OT_dict[word])
        NT_counts.append(NT_dict[word])
        i+=1
        if i >= limit:
            break

    return words, OT_counts, NT_counts
    
    
import operator
def sort_wordcounts(d: dict):
    sorted_dict = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True))
    return sorted_dict

def get_all_verses(books: list):
    old_testament_verses = []
    for i in range(N_OLD_TESTAMENT):
        old_testament_verses += books[i].verses
    new_testament_verses = []
    for i in range(N_OLD_TESTAMENT, len(books)):
        new_testament_verses += books[i].verses
    return old_testament_verses, new_testament_verses

def get_all_verses_lengths(books: list):
    old_testament_lens = []
    for i in range(N_OLD_TESTAMENT):
        old_testament_lens += books[i].get_verses_lengths()
    new_testament_lens = []
    for i in range(N_OLD_TESTAMENT, len(books)):
        new_testament_lens += books[i].get_verses_lengths()
    return old_testament_lens, new_testament_lens

def classify_verse_by_words_logodds(logodds, verse, bias=0):
    logodds_sum = 0
    for word in verse.as_tokens:
        if "'s" in word:
            print("WRONG VERSE")
            print(verse)
        try:
            logodds_sum += logodds[word]
        except:
            continue
    logodds_sum += bias
    if logodds_sum >= 0:
        return 1
    else:
        return 0

def classify_by_words_logodds(logodds, verses, bias=0):
    classifications = []
    for verse in verses:
        classifications.append( classify_verse_by_words_logodds(logodds, verse, bias) )
    return classifications

def classify_verse_by_words_logodds_with_lens(logodds, verse, ot_mean, nt_mean):
    logodds_sum = 0
    for word in verse.as_tokens:
        if "'s" in word:
            print("WRONG VERSE")
            print(verse)
        try:
            logodds_sum += logodds[word]
        except:
            continue
    if len(verse.as_string) <= nt_mean:
        logodds_sum += 1
    elif len(verse.as_string) >= ot_mean:
        logodds_sum -= 1
    if logodds_sum >= 0:
        return 1
    else:
        return 0

def classify_by_words_logodds_with_lens(logodds, verses, ot_mean, nt_mean):
    classifications = []
    for verse in verses:
        classifications.append( classify_verse_by_words_logodds_with_lens(logodds, verse, ot_mean, nt_mean) )
    return classifications

def classify_simple(verses):
    classifications = []
    for verse in verses:
        if "jesus" in verse.as_tokens or "christ" in verse.as_tokens or "apostles" in verse.as_tokens:
            classifications.append(1)
        else:
            classifications.append(0)
    return classifications

if __name__ == "__main__":
    books = parse_textfile("bible.txt")
    for book in books:
        print(book.name, len(book.sections), " sections, ", len(book.verses), " verses.")
    
    print("number of books: ", len(books))
            
    booknames = [book.name for book in books]
    testaments = ["Old Testament" if (x < N_OLD_TESTAMENT) else 'New Testament' for x in range(len(booknames)) ]
    booklengths = [book.get_text_length() for book in books]
    n_sections = [len(book.sections) for book in books]

    pd_data = {"book": booknames, "testament": testaments, "text_length": booklengths, "sections": n_sections}
    df = pd.DataFrame(data=pd_data)

    verses_lengths, average_verses_lengths_per_book = compute_verses_lengths(books)
    print("verses_lengths: ")
    print(verses_lengths)
    print("average_verse_length: ")
    print(average_verses_lengths_per_book)

    total_word_count, unique_word_count = get_total_word_count(books)
    print("total word count: ")
    print(total_word_count)
    print("unique word count? ")
    print(unique_word_count)


    OT_counts, NT_counts, all_counts = get_wordcounts(books)
    OT_counts_sorted = sort_wordcounts(OT_counts)
    NT_counts_sorted = sort_wordcounts(NT_counts)
    all_counts_sorted = sort_wordcounts(all_counts)

    print("most used words in old testament: ")
    print_dict(OT_counts_sorted, limit=30)

    print("most used words in new testament: ")
    print_dict(NT_counts_sorted, limit=30)

    print("most used words overall: ")
    print_dict(all_counts_sorted, limit=30)

    print("--------------------------------------------------------")

    f_OT_counts, f_NT_counts, f_all_counts = get_wordcounts(books, True)
    f_OT_counts_sorted = sort_wordcounts(f_OT_counts)
    f_NT_counts_sorted = sort_wordcounts(f_NT_counts)
    f_all_counts_sorted = sort_wordcounts(f_all_counts)


    # print("most used filtered words in old testament: ")
    # print_dict(f_OT_counts_sorted, limit=30)

    # print("most used filtered words in new testament: ")
    # print_dict(f_NT_counts_sorted, limit=30)

    # print("most used filtered words overall: ")
    # print_dict(f_all_counts_sorted, limit=30)

    # print("--------------------------------------------------------")
    # OT_ratio_dict = {}
    # for key in OT_counts.keys():
    #     if key not in NT_counts:
    #         OT_ratio_dict[key] = (np.inf, OT_counts[key])
    #     else:
    #         OT_ratio_dict[key] = (OT_counts[key] / NT_counts[key], OT_counts[key])
    # OT_ratio_dict_sorted = sort_wordcounts(OT_ratio_dict)

    # NT_ratio_dict = {}
    # for key in NT_counts.keys():
    #     if key not in OT_counts:
    #         NT_ratio_dict[key] = (np.inf, NT_counts[key])
    #     else:
    #         NT_ratio_dict[key] = (NT_counts[key] / OT_counts[key], NT_counts[key])
    # NT_ratio_dict_sorted = sort_wordcounts(NT_ratio_dict)

    # print("extreme evidence words for OT:")
    # count = 0
    # count_ge_3 = 0
    # for key, value in OT_ratio_dict_sorted.items():
    #     if value[0] < np.inf:
    #         break
    #     print(key, ":", value, OT_counts[key])
    #     count += 1
    #     if OT_counts[key] >= 3:
    #         count_ge_3 += 1
    # print("count of extreme evidence words for OT: ", count)
    # print("with at least 3 occurences: ", count_ge_3)

    # print("extreme evidence words for NT:")
    # count = 0
    # count_ge_3 = 0
    # for key, value in NT_ratio_dict_sorted.items():
    #     if value[0] < np.inf:
    #         break
    #     print(key, ":", value, NT_counts[key])
    #     count += 1
    #     if NT_counts[key] >= 3:
    #         count_ge_3 += 1
    # print("count of extreme evidence words for NT: ", count)
    # print("with at least 3 occurences: ", count_ge_3)

    # print("void: ")
    # print(OT_ratio_dict["void"])
    slt_books = parse_textfile("bible_slt.txt")

    kjb_ot_verses, kjb_nt_verses = get_all_verses(books)
    slt_ot_verses, slt_nt_verses = get_all_verses(slt_books)

    kjb_ot_data = [ (verse,0) for verse in kjb_ot_verses ]
    kjb_nt_data = [ (verse,1) for verse in kjb_nt_verses ]
    slt_ot_data = [ (verse,0) for verse in slt_ot_verses ]
    slt_nt_data = [ (verse,1) for verse in slt_nt_verses ]

    kjb_data = kjb_ot_data + kjb_nt_data
    slt_data = slt_ot_data + slt_nt_data

    random.shuffle(kjb_data)
    random.shuffle(slt_data)

    test_data_len = len(kjb_data) // 4
    train_data_len = len(kjb_data) - test_data_len

    kjb_train_data = kjb_data[:train_data_len]
    kjb_test_data = kjb_data[train_data_len:]
    slt_test_data = slt_data
    print("kjb_test_data len = ", len(kjb_test_data))

    OT_counts, NT_counts, all_counts = get_wordcounts_from_data(kjb_train_data, filtered=False)
    odds = get_token_NT_odds(OT_counts, NT_counts, all_counts)
    logodds = odds2logOdds(odds)

    kjb_test_verses = [ data[0] for data in kjb_test_data ]
    kjb_test_Y = np.array([ data[1] for data in kjb_test_data ])
    slt_test_verses = [ data[0] for data in slt_test_data ]
    slt_test_Y = np.array([ data[1] for data in slt_test_data ])

    kjb_logodds_predictions = np.array(classify_by_words_logodds(logodds, kjb_test_verses))
    kjb_trivial_predictions = np.zeros_like(kjb_logodds_predictions)
    kjb_simple_predictions = np.array( classify_simple(kjb_test_verses) )
    kjb_trivial_accuraccy = np.sum( kjb_test_Y == kjb_trivial_predictions ) / len(kjb_test_Y)
    kjb_simple_accuraccy = np.sum( kjb_test_Y == kjb_simple_predictions ) / len(kjb_test_Y)
    kjb_logodds_accuraccy = np.sum( kjb_test_Y == kjb_logodds_predictions ) / len(kjb_test_Y)
    print("Trivial accuracy = ", kjb_trivial_accuraccy)
    print("Simple accuracy = ", kjb_simple_accuraccy)
    print("Log Odds accuracy = ", kjb_logodds_accuraccy)

    slt_logodds_predictions = np.array(classify_by_words_logodds(logodds, slt_test_verses))
    slt_logodds_accuraccy = np.sum( slt_test_Y == slt_logodds_predictions ) / len(slt_test_Y)
    slt_trivial_predictions = np.zeros_like(slt_logodds_predictions)
    slt_simple_predictions = np.array( classify_simple(slt_test_verses) )
    slt_trivial_accuraccy = np.sum( slt_test_Y == slt_trivial_predictions ) / len(slt_test_Y)
    slt_simple_accuraccy = np.sum( slt_test_Y == slt_simple_predictions ) / len(slt_test_Y)
    print("SLT Trivial accuracy = ", slt_trivial_accuraccy)
    print("SLT Simple accuracy = ", slt_simple_accuraccy)
    print("SLT Log Odds accuracy = ", slt_logodds_accuraccy)

    OT_counts, NT_counts, all_counts = get_wordcounts_from_data(kjb_train_data, filtered=True)
    odds = get_token_NT_odds(OT_counts, NT_counts, all_counts)
    filtered_logodds = odds2logOdds(odds)
    kjb_filtered_logodds_predictions = np.array(classify_by_words_logodds(filtered_logodds, kjb_test_verses))
    kjb_filtered_logodds_accuraccy = np.sum( kjb_test_Y == kjb_filtered_logodds_predictions ) / len(kjb_test_Y)
    slt_filtered_logodds_predictions = np.array(classify_by_words_logodds(filtered_logodds, slt_test_verses))
    slt_filtered_logodds_accuraccy = np.sum( slt_test_Y == slt_filtered_logodds_predictions ) / len(slt_test_Y)
    print("KJB Log Odds accuracy filtered dataset = ", kjb_filtered_logodds_accuraccy)
    print("SLT Log Odds accuracy filtered dataset = ", slt_filtered_logodds_accuraccy)

    # logodds_predictions = np.array(classify_by_words_logodds(logodds, test_verses,bias=np.log(3/4)))
    # logodds_accuraccy = np.sum( test_Y == logodds_predictions ) / len(test_Y )
    # print("Log Odds accuracy biased 3/4 filtered dataset = ",logodds_accuraccy)
    # logodds_predictions = np.array(classify_by_words_logodds(logodds, test_verses,bias=np.log(3/4)))
    # logodds_accuraccy = np.sum( test_Y == logodds_predictions ) / len(test_Y )
    # print("SLT Log Odds accuracy biased 3/4 filtered dataset = ",logodds_accuraccy)
    
    # logodds_predictions = np.array(classify_by_words_logodds(logodds, test_verses,bias=np.log(1/4)))
    # logodds_accuraccy = np.sum( test_Y == logodds_predictions ) / len(test_Y )
    # print("Log Odds accuracy biased 1/4 filtered dataset = ",logodds_accuraccy)
    # logodds_predictions = np.array(classify_by_words_logodds(logodds, test_verses,bias=np.log(1/4)))
    # logodds_accuraccy = np.sum( test_Y == logodds_predictions ) / len(test_Y )
    # print("SLT Log Odds accuracy biased 1/4 filtered dataset = ",logodds_accuraccy)

    # logodds_predictions = np.array(classify_by_words_logodds(logodds, test_verses,bias=np.log(1/4)))
    # logodds_accuraccy = np.sum( test_Y == logodds_predictions ) / len(test_Y )
    # print("Log Odds accuracy biased 1/4 filtered dataset = ",logodds_accuraccy)
    # logodds_predictions = np.array(classify_by_words_logodds(logodds, test_verses,bias=np.log(1/4)))
    # logodds_accuraccy = np.sum( test_Y == logodds_predictions ) / len(test_Y )
    # print("SLT Log Odds accuracy biased 1/4 filtered dataset = ",logodds_accuraccy)

    old_testament_verse_lens, new_testament_verse_lens = get_all_verses_lengths(books)
    ot_mean = np.mean(old_testament_verse_lens)
    nt_mean = np.mean(new_testament_verse_lens)
    kjb_logodds_predictions = np.array(classify_by_words_logodds_with_lens(filtered_logodds, kjb_test_verses, ot_mean, nt_mean))
    kjb_logodds_accuraccy = np.sum( kjb_test_Y == kjb_logodds_predictions ) / len(kjb_test_Y)
    slt_logodds_predictions = np.array(classify_by_words_logodds_with_lens(filtered_logodds, slt_test_verses, ot_mean, nt_mean))
    slt_logodds_accuraccy = np.sum( slt_test_Y == slt_logodds_predictions ) / len(slt_test_Y)
    print("KJB Log Odds with lens accuracy filtered dataset = ", kjb_logodds_accuraccy)
    print("SLT Log Odds with lens accuracy filtered dataset = ", slt_logodds_accuraccy)

    #TODO split kbj and slt testdata