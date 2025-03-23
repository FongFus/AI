# PANDAS
import pandas as pd
#Phan Ly Thuyet
print("\nPHAN LY THUYET:")
#PANDAS
print("\nPANDAS")
# 1/ Pandas Series is essentially a one-dimensional array, equipped with an index which labels its entries. We can create a Series object, for example, by converting a list (called diameters) [4879,12104,12756,6792,142984,120536,51118,49528]
print("\nCau 1:")
data=[4879,12104,12756,6792,142984,120536,51118,49528]
diameters=pd.Series(data)
print(diameters)
# 2/ By default entries of a Series are indexed by consecutive integers, but we can specify a more meaningful index. The numbers in the above Series give diameters (in kilometers) of planets of the Solar System, so it is sensible to use names of the planet as index values:
# Index=[“Mercury”, “Venus”, “Earth”, “Mars”, “Jupyter”, “Saturn”, “Uranus”, “Neptune”]
print("\nCau 2:")
diameters=pd.Series(data, index=["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune"])
print(diameters)
# 3/ Find diameter of Earth?
print("\nCau 3:")
print(diameters["Earth"])
# 4/ Find diameters from “Mercury” to “Mars” basing on data on 2/
print("\nCau 4:")
print(diameters["Mercury":"Mars"])
# 5/  Find diameters of “Earth”, “Jupyter” and “Neptune” (with one command)?
print("\nCau 5:")
print(diameters[["Earth","Jupyter","Neptune"]])
# 6/ I want to modify the data in diameters. Specifically, I want to add the diameter of Pluto 2370. Saved the new data in the old name “diameters”.
print("\nCau 6:")
diameters["Pluton"] = 2370
print(diameters["Pluton"])
# 7/ Pandas DataFrame is a two-dimensional array equipped with one index labeling its rows, and another labeling its columns. There are several ways of creating a DataFrame. One of them is to use a dictionary of lists. Each list gives values of a column of the DataFrame, and dictionary keys give column labels:
# “diameter”=[4879,12104,12756,6792,142984,120536,51118,49528,2370]
# “avg_temp”=[167,464,15,-65,-110, -140, -195, -200, -225]
# “gravity”=[3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7]Create a pandas DataFrame, called planets.
print("\nCau 7:")
diameter = [4879, 12104, 12756, 6792, 142984, 120536, 51118, 49528, 2370]
avg_temp = [167, 464, 15, -65, -110, -140, -195, -200, -225]
gravity = [3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7]

# Tạo DataFrame
data = {
    "diameter": diameter,
    "avg_temp": avg_temp,
    "gravity": gravity
}

planets = pd.DataFrame(data)
print("\n", planets)

# 8/ Get the first 3 rows of “planets”.
print("\nCau 8:")
print(planets.head(3))
# 9/ Get the last 2 rows of “planets”.
print("\nCau 9:")
print(planets.tail(2))
# 10/ Find the name of columns of “planets”
print("\nCau 10:")
print(planets.columns)
# 11/ Since we have not specified an index for rows, by default it consists of consecutive integers. We can change it by modifying the index by using the name of the corresponding planet. Check the index after modifying.
#planets=pd.DataFrame(data, index=[])
print("\nCau 11:")
planets.index=["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune","Pluton"]
# 12/ How to get the gravity of all planets in “planets”?
print("\nCau 12:")
print(planets["gravity"])
# 13/ How to get the gravity and diameter of all planets in “planets”?
print("\nCau 13:")
print(planets[["gravity","diameter"]])
# 14/ Find the gravity of Earth using loc?
print("\nCau 14:")
print(planets.loc["Earth", "gravity"])
# 15/ Similarly, find the diameter and gravity of Earth?
print("\nCau 15:")
print(planets.loc["Earth",["diameter", "gravity"]])
# 16/ Find the gravity and diameter from Earth to Saturn?
print("\nCau 16:")
print(planets.loc["Earth":"Saturn", ["gravity", "diameter"]])
# 17/ Check (using Boolean) all the planets in “planets” that have diameter >1000?
print("\nCau 17:")
print(planets[planets["diameter"] > 1000])
# 18/ Select all planets in “planets” that have diameter>100000?
print("\nCau 18:")
print(planets[planets["diameter"] > 100000])
# 19/ Select all planets in “planets” that satisfying avg-temp>0 and gravity>5.
print("\nCau 19:")
print(planets[(planets["avg_temp"] > 0) & (planets["gravity"] > 5)])
# 20/ Sort values of diameter in “diameters” in ascending order.
print("\nCau 20:")
print(planets["diameter"].sort_values())
# 21/ Sort values of diameter in “diameters” in descending order.
print("\nCau 21:")
print(planets["diameter"].sort_values(ascending=False))
# 22/ Sort using the “gravity” column in descending order in “planets”.
print("\nCau 22:")
print(planets.sort_values(by="gravity", ascending=False))
# 23/ Sort values in the “Mercury” row.
print("\nCau 23:")
print(planets.loc["Mercury"].sort_values(ascending=False))

#SEABORNS
print("\nSEABORNS")

# 1/ Seaborn is Python library for visualizing data. Seaborn uses matplotlib to create graphics, but it provides tools that make it much easier to create several types of plots. In particular, it is simple to use seaborn with pandas dataframes.
import matplotlib.pyplot as plt
import seaborn as sns

print("\nCau 1:")
tips = sns.load_dataset("tips")
sns.set_style("whitegrid")
g = sns.lmplot(x="tip", y="total_bill", data=tips, aspect=2)
g.set_axis_labels("Tip", "Total Bill (USD)")
g.set(xlim=(0, 10), ylim=(0, 100))
plt.title("Cau 1:")
plt.show()

# 2/ Display name of datasets.
print("\nCau 2:")
print(sns.get_dataset_names())
# 3/ How can get a pandas dataframe with the data.
print("\nCau 3:")
print(sns.load_dataset("tips"))
# 4/ How to produce a scatter plot showing the bill amount on the x axis and the tip amount on the y axis?
print("\nCau 4:")
sns.scatterplot(x='total_bill', y='tip',data=tips)
plt.xlabel('Total Bill Amount ($)')
plt.ylabel('Tip Amount ($)')
plt.title('Cau 4:')
plt.show()
# 5/ By default, seaborn uses the original matplotlib settings for fonts, colors etc. How to modify font=1.2 and color=darkgrid?
print("\nCau 5:")
sns.set_context("talk", font_scale=1.2)
sns.set_style("darkgrid")
tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.title('Cau 5:')
plt.show()
# 6/ We can use the values in the “day” column to assign marker colors. How?
print("\nCau 6:")
tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips, hue="day")
plt.title('Cau 6:')
plt.show()
# 7/ Next, we set different marker sizes based on values in the “size” column.
print("\nCau 7:")
tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips, hue="day", size="size")
plt.title('Cau 7:')
plt.show()
# 8/ We can also split the plot into subplots based on values of some column. Below we create two subplots, each displaying data for a different value of the “time” column
print("\nCau 8:")
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="time")
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()
# 9/ We can subdivide the plot even further using values of the “sex” column
print("\nCau 9:")
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="time", row="sex")
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()

#Phan Thuc Hanh
print("\nPHAN THUC HANH:")
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
