{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages = with pkgs; [ pkgs.git pyright ruff python3Packages.matplotlib python3Packages.tkinter ];

  languages.python = {
    enable = true;
    poetry = {
      enable = true;
      activate.enable = true;
    };
  };
}
