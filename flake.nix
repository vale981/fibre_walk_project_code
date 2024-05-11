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
        poetry2nix = inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs;};
      in
        {
          packages = {
            fibreMagic = poetry2nix.mkPoetryApplication {
              projectDir = self;
              preferWheels = true;


              overrides = poetry2nix.overrides.withDefaults (final: prev: {
                matplotlib = with pkgs; prev.matplotlib.overridePythonAttrs (
                  old:
                  let
                    interactive = true;

                    passthru = {
                      config = {
                        directories = { basedirlist = "."; };
                        libs = {
                          system_freetype = true;
                          system_qhull = true;
                        };
                      };
                    };

                    inherit (pkgs) tk tcl wayland qhull;
                    inherit (pkgs.xorg) libX11;
                  in
                    {
                      XDG_RUNTIME_DIR = "/tmp";

                      buildInputs = old.buildInputs or [ ] ++ [
                        pkgs.which
                      ];

                      propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [
                        final.certifi
                        pkgs.libpng
                        pkgs.freetype
                        qhull
                      ]
                      ++ [ pkgs.cairo pkgs.librsvg final.pycairo pkgs.gtk3 pkgs.gtk4 pkgs.gobject-introspection final.pygobject3 ]  ;

                      nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                        pkg-config
                      ] ++ [
                        final.setuptools-scm
                      ];

                      # Clang doesn't understand -fno-strict-overflow, and matplotlib builds with -Werror
                      hardeningDisable = if stdenv.isDarwin then [ "strictoverflow" ] else [ ];

                      passthru = old.passthru or { } // passthru;

                      MPLSETUPCFG = pkgs.writeText "mplsetup.cfg" (lib.generators.toINI { } passthru.config);

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
              ruff pyright
            ];
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
