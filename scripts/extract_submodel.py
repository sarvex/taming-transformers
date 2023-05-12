import torch
import sys

if __name__ == "__main__":
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    submodel = sys.argv[3] if len(sys.argv) > 3 else "cond_stage_model"
    print(f"Extracting {submodel} from {inpath} to {outpath}.")

    sd = torch.load(inpath, map_location="cpu")
    new_sd = {
        "state_dict": {
            k.split(".", 1)[-1]: v
            for k, v in sd["state_dict"].items()
            if k.startswith("cond_stage_model")
        }
    }
    torch.save(new_sd, outpath)
