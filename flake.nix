{
    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = {nixpkgs, flake-utils, ... }: 
    flake-utils.lib.eachSystem flake-utils.lib.allSystems (system:
    let
        pkgs = import nixpkgs {
            inherit system;
        };
    in rec {
        devShell = (pkgs.buildFHSUserEnv {
            name = "py_env";
            targetPkgs = pkgs: (with pkgs; [
            nodePackages.pyright
                (python312.withPackages(ps: with ps; [
                    virtualenv
                ]))
            ]);
            profile = ''
                if [ ! -d "venv" ]; then
                    virtualenv venv
                    source venv/bin/activate
                    pip install -r requirements.txt
                else
                    source venv//bin/activate
                fi
            '';
            runScript = "bash";
        }).env;
    });
}
