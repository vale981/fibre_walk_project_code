{
  description = "Analysis code for the realization of the NHQW in fibre loops.";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        poetry2nix = inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
      in
      {
        packages = {
          fibreMagic = poetry2nix.mkPoetryApplication {
            projectDir = self;
            preferWheels = true;

            overrides = poetry2nix.overrides.withDefaults (final: prev: {
              matplotlib = with pkgs; prev.matplotlib.override (
                {
                  enableGtk3 = true;
                }
              );
            });
          };
          default = self.packages.${system}.fibreMagic;
        };

        # Shell for app dependencies.
        #
        #     nix develop
        #
        # Use this shell for developing your app.
        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.fibreMagic ];
          package = with pkgs; [
            ruff
            pyright
            python3Packages.jupyter
          ];

          shellHook = ''
          export PYTHONPATH=$(pwd)/src:$PYTHONPATH
          '';
        };

        # Shell for poetry.
        #
        #     nix develop .#poetry
        #
        # Use this shell for changes to pyproject.toml and poetry.lock.
        devShells.poetry = pkgs.mkShell {
          packages = [ pkgs.poetry ];
        };
      });
}
