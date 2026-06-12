import re

file_path = "/Users/yashpatil/Local_Mark1/Xariv/SSRN Paper/paper_final_v4.md"
with open(file_path, "r") as f:
    text = f.read()

# Abstract
text = text.replace("Bull 593 days", "Bull 638 days")
text = text.replace("Sideways 955 days", "Sideways 968 days")
text = text.replace("Bear 121 days", "Bear 123 days")
text = text.replace("Sharpe ratio 1.667, t = 3.24", "Sharpe ratio 1.726, t = 3.34")
text = text.replace("Sharpe ratio 1.485, t = 2.28", "Sharpe ratio 1.458, t = 2.25")
text = text.replace("Sharpe ratio 2.26", "Sharpe ratio 1.46")
text = text.replace("n = 121 days", "n = 123 days")
text = text.replace("(F ≈ 5, p ≈ 0.009)", "(F ≈ 2.4, p ≈ 0.096)")
text = text.replace("Sharpe 0.794", "Sharpe 0.800")

# Sample Alignment
text = text.replace("Bull 593 days (35.5%), Sideways 955 days (57.2%), Bear 121 days (7.3%).", 
                    "Bull 638 days (36.9%), Sideways 968 days (56.0%), Bear 123 days (7.1%).")

# Table 2 and Section 4 text
text = text.replace("Bull | MKT | 593 | 23.59 | 15.89 | 1.485 | 2.28 | 0.023 | *", "Bull | MKT | 600 | 23.30 | 15.98 | 1.458 | 2.25 | 0.024 | *")
text = text.replace("Bull | SMB | 593 | 0.61 | 11.70 | 0.052 | 0.08 | 0.936 |", "Bull | SMB | 600 | 2.17 | 11.83 | 0.184 | 0.28 | 0.780 |")
text = text.replace("Bull | HML | 593 | 9.80 | 13.75 | 0.712 | 1.09 | 0.275 |", "Bull | HML | 600 | 13.22 | 13.95 | 0.948 | 1.46 | 0.145 |")
text = text.replace("Bull | WML | 593 | 10.60 | 12.44 | 0.852 | 1.31 | 0.192 |", "Bull | WML | 600 | 10.22 | 12.82 | 0.797 | 1.23 | 0.219 |")

text = text.replace("Sideways | MKT | 955 | 1.65 | 12.14 | 0.136 | 0.27 | 0.791 |", "Sideways | MKT | 946 | 0.47 | 12.06 | 0.039 | 0.08 | 0.936 |")
text = text.replace("Sideways | SMB | 955 | 3.10 | 11.15 | 0.278 | 0.54 | 0.588 |", "Sideways | SMB | 946 | 0.82 | 11.14 | 0.074 | 0.14 | 0.889 |")
text = text.replace("Sideways | HML | 955 | 6.15 | 11.58 | 0.531 | 1.03 | 0.302 |", "Sideways | HML | 946 | 6.08 | 11.55 | 0.526 | 1.02 | 0.308 |")
text = text.replace("Sideways | WML | 955 | 20.77 | 12.46 | 1.667 | 3.24 | 0.001 | **", "Sideways | WML | 946 | 21.21 | 12.29 | 1.726 | 3.34 | 0.001 | ***")

text = text.replace("Bear | MKT | 121 | 37.55 | 37.43 | 1.003 | 0.70 | 0.488 |", "Bear | MKT | 123 | 46.19 | 37.02 | 1.248 | 0.87 | 0.386 |")
text = text.replace("Bear | SMB | 121 | −32.91 | 21.66 | −1.520 | −1.05 | 0.294 |", "Bear | SMB | 123 | −22.25 | 21.25 | −1.047 | −0.73 | 0.467 |")
text = text.replace("Bear | HML | 121 | 45.29 | 20.07 | 2.256 | 1.56 | 0.121 |", "Bear | HML | 123 | 28.26 | 19.41 | 1.456 | 1.02 | 0.310 |")
text = text.replace("Bear | WML | 121 | −12.89 | 24.06 | −0.536 | −0.37 | 0.711 |", "Bear | WML | 123 | −13.28 | 23.62 | −0.562 | −0.39 | 0.697 |")

text = text.replace("Bull regime (n = 593)", "Bull regime (n = 600)")
text = text.replace("23.59% annualised (SR = 1.485, t = 2.28*)", "23.30% annualised (SR = 1.458, t = 2.25*)")
text = text.replace("(WML: 10.60%, SR = 0.852)", "(WML: 10.22%, SR = 0.797)")
text = text.replace("Sideways regime (n = 955)", "Sideways regime (n = 946)")
text = text.replace("20.77% annualised (SR = 1.667, t = 3.24**)", "21.21% annualised (SR = 1.726, t = 3.34***)")
text = text.replace("estimated SR = 1.54, t = 2.98**", "estimated SR = 1.571, t = 3.01**")
text = text.replace("MKT is near zero (1.65%)", "MKT is near zero (0.47%)")
text = text.replace("Bear regime (n = 121)", "Bear regime (n = 123)")
text = text.replace("(45.29% annualised, SR = 2.256)", "(28.26% annualised, SR = 1.456)")
text = text.replace("(−32.91%, SR = −1.520)", "(−22.25%, SR = −1.047)")
text = text.replace("(37.55%, SR = 1.003)", "(46.19%, SR = 1.248)")
text = text.replace("(−12.89%, SR = −0.536)", "(−13.28%, SR = −0.562)")
text = text.replace("(spread = 2.20", "(spread = 2.288")
text = text.replace("(spread = 1.35", "(spread = 1.419")

# Table 3 Panel A
text = text.replace("Bull | 0.47 | 0.491 | 1 | 615", "Bull | 0.97 | 0.380 | 2 | 621")
text = text.replace("Sideways | 1.00 | 0.367 | 2 | 933", "Sideways | 1.37 | 0.254 | 2 | 925")
text = text.replace("Bear | 1.43 | 0.234 | 1 | 120", "Bear | 2.31 | 0.132 | 1 | 122")

# Table 3 Panel B
text = text.replace("Bull | MKT | 0.47 | 0.491 | 0.567 | 1 | 615 | No", "Bull | MKT | 0.97 | 0.380 | 0.556 | 2 | 621 | No")
text = text.replace("Bull | SMB | 2.65 | 0.071 | 0.212 | 2 | 615 | No", "Bull | SMB | 2.12 | 0.121 | 0.424 | 2 | 621 | No")
text = text.replace("Bull | HML | 2.06 | 0.084 | 0.212 | 4 | 615 | No", "Bull | HML | 1.35 | 0.250 | 0.424 | 4 | 621 | No")
text = text.replace("Bull | RMW | 1.51 | 0.222 | 0.390 | 2 | 615 | No", "Bull | RMW | 0.31 | 0.735 | 0.788 | 2 | 621 | No")
text = text.replace("Bull | CMA | 1.98 | 0.080 | 0.212 | 5 | 615 | No", "Bull | CMA | 1.60 | 0.203 | 0.424 | 2 | 621 | No")

text = text.replace("Sideways | MKT | 1.00 | 0.367 | 0.501 | 2 | 933 | No", "Sideways | MKT | 1.37 | 0.254 | 0.424 | 2 | 925 | No")
text = text.replace("Sideways | SMB | 2.06 | 0.085 | 0.212 | 4 | 933 | No", "Sideways | SMB | 1.54 | 0.202 | 0.424 | 3 | 925 | No")
text = text.replace("Sideways | HML | 0.75 | 0.556 | 0.595 | 4 | 933 | No", "Sideways | HML | 0.22 | 0.953 | 0.953 | 5 | 925 | No")
text = text.replace("Sideways | RMW | 0.89 | 0.409 | 0.512 | 2 | 933 | No", "Sideways | RMW | 0.81 | 0.445 | 0.556 | 2 | 925 | No")
text = text.replace("Sideways | CMA | 1.21 | 0.305 | 0.458 | 3 | 933 | No", "Sideways | CMA | 0.63 | 0.429 | 0.556 | 1 | 925 | No")

text = text.replace("Bear | MKT | 1.43 | 0.234 | 0.390 | 1 | 120 | No", "Bear | MKT | 2.31 | 0.132 | 0.424 | 1 | 122 | No")
text = text.replace("Bear | SMB | 8.21 | 0.005 | 0.071 | 1 | 120 | No", "Bear | SMB | 5.68 | 0.019 | 0.204 | 1 | 122 | No")
text = text.replace("Bear | HML | 0.51 | 0.726 | 0.726 | 4 | 120 | No", "Bear | HML | 0.61 | 0.654 | 0.755 | 4 | 122 | No")
text = text.replace("Bear | RMW | 1.73 | 0.134 | 0.287 | 5 | 120 | No", "Bear | RMW | 1.39 | 0.235 | 0.424 | 5 | 122 | No")
text = text.replace("Bear | CMA | 6.98 | 0.009 | 0.071 | 1 | 120 | No", "Bear | CMA | 2.85 | 0.027 | 0.204 | 4 | 122 | No")

text = text.replace("Bear × SMB (F = 8.21, p_raw = 0.005)", "Bear × SMB (F = 5.68, p_raw = 0.019)")
text = text.replace("Bear × CMA (F = 6.98, p_raw = 0.009)", "Bear × CMA (F = 2.85, p_raw = 0.027)")
text = text.replace("AUC-ROC = 0.548 and McFadden\nR² = 0.004", "AUC-ROC = 0.546 and McFadden\nR² = 0.004")

# Panel D
text = text.replace("Bear | 5.06 (0.008) | 4.89 (0.009) | 2 |", "Bear | 2.39 (0.096) | 1.94 (0.149) | 2 |")
text = text.replace("purchases (F = 5.06, p = 0.008, lag = 2)", "purchases (F = 2.39, p = 0.096, lag = 2)")
text = text.replace("sales (F = 4.89, p = 0.009, lag = 2)", "sales (F = 1.94, p = 0.149, lag = 2)")

# Strategy table
text = text.replace("Regime-Aware | 65.7 | 10.75 | 13.54 | 0.794 | 1.103 | −17.5 | 0.615 | 1.80 |", 
                    "Regime-Aware | 66.5 | 10.85 | 13.55 | 0.800 | 1.119 | −17.5 | 0.621 | 1.81 |")
text = text.replace("Sharpe ratio by 0.061", "Sharpe ratio by 0.067")

# Appendix A2 (Just update the counts since SMB/HML correlations don't change text, but let's check counts)
text = text.replace("Bull | MKT | 632", "Bull | MKT | 638")
text = text.replace("Bull | SMB | 632", "Bull | SMB | 638")
text = text.replace("Bull | HML | 632", "Bull | HML | 638")
text = text.replace("Bull | RMW | 632", "Bull | RMW | 638")
text = text.replace("Bull | CMA | 632", "Bull | CMA | 638")
text = text.replace("Sideways | MKT | 976", "Sideways | MKT | 968")
text = text.replace("Sideways | SMB | 976", "Sideways | SMB | 968")
text = text.replace("Sideways | HML | 976", "Sideways | HML | 968")
text = text.replace("Sideways | RMW | 976", "Sideways | RMW | 968")
text = text.replace("Sideways | CMA | 976", "Sideways | CMA | 968")
text = text.replace("Bear | MKT | 121", "Bear | MKT | 123")
text = text.replace("Bear | SMB | 121", "Bear | SMB | 123")
text = text.replace("Bear | HML | 121", "Bear | HML | 123")
text = text.replace("Bear | RMW | 121", "Bear | RMW | 123")
text = text.replace("Bear | CMA | 121", "Bear | CMA | 123")

# Additional paragraphs
if "Filtered probabilities are computed via an explicit forward recursion" not in text:
    text = text.replace("DCC Bear/Bull correlation means and t-stat;", "DCC Bear/Bull correlation means and t-stat; Filtered probabilities are computed via an explicit forward recursion; we note that hmmlearn's posterior utilities return smoothed (forward-backward) probabilities, which would reintroduce look-ahead.")
    text = text.replace("The DCC average pairwise correlation rises\nsignificantly in Bear regimes (mean 0.30 vs. 0.25 in Bull)", "The DCC average pairwise correlation rises\nsignificantly in Bear regimes (mean 0.30 vs. 0.25 in Bull). Filtered probabilities are computed via an explicit forward recursion; we note that hmmlearn's posterior utilities return smoothed (forward-backward) probabilities, which would reintroduce look-ahead.")

if "HMM parameters (transition matrix, emission moments) are estimated once over the full sample" not in text:
    text = text.replace("## 9. Limitations", "## 9. Limitations\n\n**HMM Estimation.** HMM parameters (transition matrix, emission moments) are estimated once over the full sample; conditional on these parameters, state probabilities are filtered. Fully out-of-sample labeling would require expanding-window re-estimation, which we leave as an extension.")


with open(file_path, "w") as f:
    f.write(text)

print("Done updating paper")
