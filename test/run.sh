run_adme -p Caco2_Wang -t train -d maccs -lr 0.01 --smiles_col Drug --target_col Y --batch 16 --epochs 5
run_adme -p Caco2_Wang -t tune -d maccs -lr 0.01 --smiles_col Drug --batch 16 --epochs 5 -pd models/maccs.pth -data dataset/caco2_valid.csv
run_adme -p Caco2_Wang -t predict -d maccs --smiles_col Drug -pd models/maccs-tuned.pth -data dataset/caco2_test.csv
