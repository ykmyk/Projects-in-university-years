LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"
DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(),
                              LETTERS_NODIA + LETTERS_NODIA.upper())

gold_words = open("test_gold.txt", encoding="utf-8-sig").read().translate(DIA_TO_NODIA)  #.split()
print(gold_words)

# system_words = open("system.txt", encoding="utf-8-sig").read().translate(DIA_TO_NODIA).split()

# for i, (g, s) in enumerate(zip(gold_words, system_words)):
#     if g != s:
#         print(f"First difference at token {i}:")
#         print(f"  GOLD:   {repr(g)}")
#         print(f"  SYSTEM: {repr(s)}")
#         break

# print("Total tokens:", len(gold_words), len(system_words))
