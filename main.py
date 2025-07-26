import argparse
import os
from ast import literal_eval


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=str, choices=["gpt2_alloy", "gpt2_ionic", "crygen"], required=False)
    parser.add_argument("-c", "--config", type=str, required=False, default="")
    parser.add_argument("-cg", "--config_gpt2", type=str, required=False, default="")
    parser.add_argument("-cc", "--config_crygen", type=str, required=False, default="default")
    parser.add_argument("-d", "--dataset", type=str, required=False, default="")
    parser.add_argument("-g", "--generate", type=str, choices=["gpt2_alloy", "gpt2_ionic", "crygen_alloy", "crygen_ionic"], required=False)
    parser.add_argument("-b", "--generate_batch_size", type=int, required=False, default=4)
    parser.add_argument("-nf", "--n_formula", type=int, required=False, default=1)
    parser.add_argument("-ns", "--n_structure_per_formula", type=int, required=False, default=4)
    parser.add_argument("-s", "--save_path", type=str, required=False, default="./cif_tmp")
    parser.add_argument("-f", "--formula", type=str, help="if the formula is provided, the workflow will generate structures based on the formula", required=False, default=[], nargs="+")
    parser.add_argument("-p", "--prop_dict", type=literal_eval, required=False, default="{}")
    args = parser.parse_args()

    if args.train is not None and args.generate is not None:
        raise ValueError("Cannot train and generate at the same time")

    if args.train is not None:
        from formula_gen import GPT2_chem
        if args.train == "gpt2_alloy":
            GPT2_chem.train("alloy", args.prop_dict, args.config, args.dataset)
        elif args.train == "gpt2_ionic":
            GPT2_chem.train("ionic", args.prop_dict, args.config, args.dataset)
        elif args.train == "crygen":
            from CRYGEN_model.train import train as crygen_train
            crygen_train(args.config)
    
    if args.generate is not None:
        os.makedirs(name=args.save_path, exist_ok=True)
        from formula_gen import GPT2_chem
        if args.generate == "gpt2_alloy":
            print(GPT2_chem.generate_formula("alloy", args.n_formula, args.prop_dict, config_file=args.config_gpt2, dataset_file=args.dataset))
        elif args.generate == "gpt2_ionic":
            print(GPT2_chem.generate_formula("ionic", args.n_formula, args.prop_dict, config_file=args.config_gpt2, dataset_file=args.dataset))
        elif args.generate.startswith("crygen"):
            from CRYGEN_model.inference import inference as crygen_inference
            if not args.formula:
                if args.generate == "crygen_alloy":
                    formulas = GPT2_chem.generate_formula("alloy", args.n_formula, args.prop_dict, config_file=args.config_gpt2, dataset_file=args.dataset)
                elif args.generate == "crygen_ionic":
                    formulas = GPT2_chem.generate_formula("ionic", args.n_formula, args.prop_dict, config_file=args.config_gpt2, dataset_file=args.dataset)
                else:
                    raise ValueError(f"Invalid generate type: {args.generate}")
                print("generated formulas: ", formulas)
            else:
                formulas = args.formula
            crygen_inference(args.save_path, args.config_crygen, formulas, args.prop_dict, args.n_structure_per_formula, args.generate_batch_size)
