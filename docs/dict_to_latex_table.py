import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-c","--caption",type=str,required=False,help="Caption of the table")
parser.add_argument("-t","--tab",type=int,required=False,default=4,help="number of spaces per tab")

args = parser.parse_args()

TAB = args.tab*" "
n_columns = 3


d1 = {'Source_2_animagineXL3': 0.800000011920929,
 'Source_29_SDXL': 0.5076923370361328,
 'Source_7_gigaGan': 0.34567901492118835,
 'Copilot_images': 0.9605262875556946,
 'Source_32_Dreamshaper': 0.921875,
 'Source_9_10_kandinsky': 0.7843137383460999,
 'LongCaptions': 0.8068181872367859,
 'Source_25_SD1_5': 0.3650793731212616,
 'Lexica_images': 0.9145299196243286,
 'Source_6_dreamlike': 0.6379310488700867,
 'MidJourneyV6': 0.8947368264198303,
 'Leonardo_images': 0.72826087474823,
 'Source_27_SD2_1': 0.4893617033958435,
 'Source_30_31_stylegan2_3': 0.12087912112474442,
 'Source_17_pixartAlpha': 0.8309859037399292,
 'null': 0.9373825788497925,
 'Source_21_playground': 0.859375,
 'Source_18_pixartSigma': 0.78125,
 'Source_4_DF-XL': 0.1358024626970291,
 'Ideogram_images': 0.8051947951316833}

d2 = {'Source_2_animagineXL3': 1.0,
 'Source_29_SDXL': 0.9384615421295166,
 'Source_7_gigaGan': 0.8888888955116272,
 'Copilot_images': 1.0,
 'Source_32_Dreamshaper': 0.96875,
 'Source_9_10_kandinsky': 1.0,
 'LongCaptions': 0.9431818127632141,
 'Source_25_SD1_5': 0.920634925365448,
 'Lexica_images': 0.9743589758872986,
 'Source_6_dreamlike': 0.982758641242981,
 'MidJourneyV6': 0.9736841917037964,
 'Leonardo_images': 0.8804348111152649,
 'Source_27_SD2_1': 0.957446813583374,
 'Source_30_31_stylegan2_3': 0.9890109896659851,
 'Source_17_pixartAlpha': 0.98591548204422,
 'null': 0.9768315553665161,
 'Source_21_playground': 0.953125,
 'Source_18_pixartSigma': 1.0,
 'Source_4_DF-XL': 0.9506173133850098,
 'Ideogram_images': 0.9740259647369385}

d3 = dict()

for gen in d1:
   d3[gen] = [d1[gen],d2[gen]]

table = """\\begin{table} 
\\centering\n"""
table += TAB + "\\begin{tabular}{|" + n_columns * "c|" + "}\n"
table += 2*TAB + "\\hline\n"
table += 2*TAB + "Generator & Before fine-tuning & After fine-tuning\\\\\n"
table += 2*TAB + "\\hline\n"
for gen in d3:
   table += 2*TAB + gen.replace("_",r"\_") + " & "
   for i in range(len(d3[gen])):
      if i+1 < len(d3[gen]):
         table += f"{float(d3[gen][i]):.3f} & "
      else:
         table += f"{float(d3[gen][i]):.3f}" + " \\\\\n"
   table += 2*TAB + "\\hline\n"

table += TAB + "\\end{tabular}\n"
if args.caption:
    table += TAB + "\\caption{" + args.caption + "}\n"

table += "\\end{table}"

print(table)