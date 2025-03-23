# PANDAS
import pandas as pd

# Đọc file TSV (Tab-Separated Values)
df = pd.read_csv("04_gap-merged.tsv", sep="\t")

# 1/ Show the first 5 lines of tsv file.
print("\nCau 1:")
print(df.head(5))

# 2/ Find the number of row and column of this file.
print("\nCau 2:")
num_rows, num_cols = df.shape
print(f"Số dòng: {num_rows}, Số cột: {num_cols}")

# 3/ Print the name of the columns.
print("\nCau 3:")
print(df.columns)

# 4/ What is the type of the column names?
print("\nCau 4:")
print(type(df.columns))

# 5/ Get the country column and save it to its own variable. Show the first 5 observations.
print("\nCau 5:")
country_col = df["country"]
print(country_col.head(5))

# 6/ Show the last 5 observations of this column.
print("\nCau 6:")
print(country_col.tail(5))

# 7/ Look at country, continent and year. Show the first 5 observations of these columns, and the last 5 observations.
print("\nCau 7:")
print(df[["country", "continent", "year"]].head(5))
print(df[["country", "continent", "year"]].tail(5))

#
# PANDAS
# 8/ How to get the first row of tsv file? How to get the 100th row.
print("\nCau 8:")
print(df.iloc[0])      # Dòng đầu tiên
print(df.iloc[99])     # Dòng thứ 100 (vì index bắt đầu từ 0)

# 9/ Try to get the first column by using a integer index. And get the first and last column by passing the integer index.
print("\nCau 9:")
print(df.iloc[:, 0])          # Cột đầu tiên
print(df.iloc[:, [0, -1]])    # Cột đầu tiên và cột cuối cùng

# 10/ How to get the last row with .loc? Try with index -1? Correct?
print("\nCau 10:")
print(df.loc[df.index[-1]])  # Dòng cuối cùng

# 11/ How to select the first, 100th, 1000th rows by two methods?
print("\nCau 11:")
print(df.iloc[[0, 99, 999]])
print(df.loc[df.index[[0, 99, 999]]])

# 12/ Get the 43rd country in our data using .loc, .iloc?
print("\nCau 12:")
print(df.iloc[42]["country"])
print(df.loc[df.index[42], "country"])

# 13/ How to get the first, 100th, 1000th rows from the first, 4th and 6thcolumns?
print("\nCau 13:")
print(df.iloc[[0, 99, 999], [0, 3, 5]])

#
# PANDAS
# 14/ Get first 10 rows of our data (tsv file)?
print("\nCau 14:")
print(df.head(10))

# 15/ For each year in our data, what was the average life expectation?
print("\nCau 15:")
avg_life_expectancy = df.groupby("year")["lifeExp"].mean()
print(avg_life_expectancy)

# 16/ Using subsetting method for the solution of 15/?
print("\nCau 16:")
for year in df["year"].unique():
    avg = df[df["year"] == year]["lifeExp"].mean()
    print(f"Năm {year}: {avg:.2f}")

# 17/ Create a series with index 0 for ‘banana’ and index 1 for ’42’?
print("\nCau 17:")
s = pd.Series(["banana", 42], index=[0, 1])
print(s)

# 18/ Similar to 17, but change index ‘Person’ for ‘Wes MCKinney’ and index ‘Who’ for ‘Creator of Pandas’?
print("\nCau 18:")
s = pd.Series(["Wes McKinney", "Creator of Pandas"], index=["Person", "Who"])
print(s)

# 19/ Create a dictionary for pandas with the data as ‘Occupation’: [’Chemist’, ’Statistician’], ’Born’: [’1920-07-25’, ’1876-06-13’],’Died’: [’1958-04-16’, ’1937-10-16’],’Age’: [37, 61] and the index is ‘Franklin’,’Gosset’ with four columns as indicated.
print("\nCau 19:")
data = {
    "Occupation": ["Chemist", "Statistician"],
    "Born": ["1920-07-25", "1876-06-13"],
    "Died": ["1958-04-16", "1937-10-16"],
    "Age": [37, 61]
}
df_people = pd.DataFrame(data, index=["Franklin", "Gosset"])
print(df_people)
