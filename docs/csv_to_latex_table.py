import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n","--name",type=str,required=True,help="Path to the csv file to convert into a latex table")
parser.add_argument("-c","--caption",type=str,required=False,help="Caption of the table")
parser.add_argument("-t","--tab",type=int,required=False,default=4,help="number of spaces per tab")

args = parser.parse_args()

TAB = args.tab*" "

table = """\\begin{table} 
\\centering\n"""
table += TAB + "\\begin{tabular}" 
with open(args.name) as f:
    content = f.read().split("\n")
    content = [line.split(",") for line in content]
    n_columns = len(content[0])
    table += "{|" + n_columns*"c|" + "}\n"
    table += 2*TAB + "\\hline\n"
    for i in range(len(content)):
        for j in range(n_columns):
            try:
                value = float(content[i][j])
                string = f"{value:.3f}"
            except ValueError:
                string = content[i][j]
                if "_" in string:
                    string = string.replace("_",r"\_")
            if j == 0:
                table += 2*TAB + string + " & "
            elif j+1 == n_columns:
                table +=  string + "\\\\\n"
                table += 2*TAB + "\\hline\n"
            else:
                table += string + " & "

table += TAB + "\\end{tabular}\n"
if args.caption:
    table += TAB + "\\caption{" + args.caption + "}\n"

table += "\\end{table}"

print(table)