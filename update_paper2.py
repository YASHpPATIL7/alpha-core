import re

file_path = "/Users/yashpatil/Local_Mark1/Xariv/SSRN Paper/paper_final_v4.md"
with open(file_path, "r") as f:
    text = f.read()

text = text.replace("(F = 6.98, p_raw = 0.009)", "(F = 2.85, p_raw = 0.027)")
text = text.replace("The Bear regime sample (n = 121) is", "The Bear regime sample (n = 123) is")
text = text.replace("covers only 121 trading days", "covers only 123 trading days")
text = text.replace("(Sharpe 1.667, t = 3.24**)", "(Sharpe 1.726, t = 3.34***)")
text = text.replace("(Sharpe 1.485, t = 2.28*)", "(Sharpe 1.458, t = 2.25*)")

# Fix Bear * SMB raw F=8.21 which was on the same line as CMA
text = text.replace("Bear × SMB (F = 8.21, p_raw = 0.005) and Bear × CMA", "Bear × SMB (F = 5.68, p_raw = 0.019) and Bear × CMA")

with open(file_path, "w") as f:
    f.write(text)

print("Second pass done")
