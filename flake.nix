{
  description = "colibrì — a pure-C inference engine for large MoE and dense models, experts streamed from disk";

  # flake.lock (committed) pins these branch inputs to exact commit SHAs,
  # so builds are reproducible; refresh with `nix flake update`.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};

        # Python with the packages needed by `coli`: the offline converter tools
        # (torch, safetensors, huggingface-hub, numpy), the benchmarks (tokenizers,
        # datasets) and pictures through the gateway (pillow: pyproject's `vision` extra).
        pythonEnv = pkgs.python3.withPackages (
          ps:
            with ps; [
              torch
              safetensors
              huggingface-hub
              numpy
              pillow
              tokenizers
              datasets
            ]
        );

        isDarwin = pkgs.stdenv.hostPlatform.isDarwin;

        # Real version from the engine's single source of truth (c/version.py).
        colibriVersion = let
          m = builtins.match ''.*__version__ = "([^"]+)".*'' (builtins.readFile ./c/version.py);
        in
          if m == null
          then "0"
          else builtins.head m;

        # Apple clang has no OpenMP runtime and the Makefile only finds libomp via
        # `brew`, absent here; expose omp.h + libomp as OMPDIR or macOS goes single-threaded.
        colibriOmp = pkgs.symlinkJoin {
          name = "colibri-openmp";
          paths = [pkgs.llvmPackages.openmp pkgs.llvmPackages.openmp.dev];
        };

        # Portable default per arch — never -mcpu=native, which would pin a
        # distributed binary to the builder's core and break substitution.
        archBaseline =
          if pkgs.stdenv.hostPlatform.isx86_64
          then "x86-64-v3"
          else if isDarwin
          then "" # arm64 macOS: NEON is baseline
          else "armv8-a";

        # Build args: portable ARCH, plus (on macOS) OpenMP via OMPDIR and the
        # Metal backend — Apple clang provides neither on its own.
        buildArgs =
          "ARCH=${archBaseline}"
          + pkgs.lib.optionalString isDarwin " OMPDIR=${colibriOmp} METAL=1";

        # `vulkan = true` builds every engine with VK=1 (Linux). The loader is opened
        # at run time (docs/vulkan.md), so the build needs only the headers and glslc.
        mkColibri = {vulkan ? false}: let
          makeArgs = buildArgs + pkgs.lib.optionalString vulkan " VK=1";
        in
          pkgs.stdenv.mkDerivation {
            pname =
              if vulkan
              then "colibri-vulkan"
              else "colibri";
            version = colibriVersion;
            src = ./.;

            nativeBuildInputs = with pkgs;
              [makeWrapper]
              ++ lib.optionals vulkan [shaderc vulkan-headers];

            # Compiler comes from stdenv (clang on Darwin, gcc on Linux); these add
            # only the extra build/runtime libs each platform needs.
            buildInputs = with pkgs;
              lib.optionals stdenv.hostPlatform.isDarwin [
                llvmPackages.openmp # libomp runtime for the OpenMP build
                apple-sdk_15 # SDK 15 headers enable the Metal residency-set path
              ]
              ++ lib.optionals stdenv.hostPlatform.isLinux [
                stdenv.cc.cc.lib # libgomp.so.1 in the runtime closure
              ];

            buildPhase = ''
              runHook preBuild
              # `make install` builds and stages every engine (colibri, glm53, inkling,
              # kimi_k3, olmoe, qwen36, qwen38, deepseek_v41, mimo, qwenimage, laya,
              # gliner_decide, the TUI, and deepseek_v4 where COLI_V4_SUPPORTED) and
              # the Python modules beside coli under $out/lib/colibri, so coli's
              # HERE-relative engine_for() finds each.
              make -C c install ${makeArgs} \
                DESTDIR=$out PREFIX= BINDIR=/bin LIBEXECDIR=/lib/colibri
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall

              # `make install` handled the engines, coli and the support
              # modules (v4_dsml.py and v41_dsml.py among them); fixups remain:

              # 1. `make install`'s file lists miss tools/iq3xxs_grid.json
              #    (iq3_pack.py loads it for `coli convert --xbits e8`, with no
              #    fallback). Back it in — the guard defers to a future
              #    `make install` that stages it itself.
              [ -e "$out/lib/colibri/tools/iq3xxs_grid.json" ] || \
                install -m 644 c/tools/iq3xxs_grid.json "$out/lib/colibri/tools/"
              ${pkgs.lib.optionalString vulkan ''
                # The Vulkan backend looks for shaders/qmatmul.spv next to the
                # engine binary (backend_vulkan.c, coli_vk_shader_path); the
                # build leaves the .spv in c/shaders and `make install` skips them.
                install -d "$out/lib/colibri/shaders"
                install -m 644 c/shaders/*.spv "$out/lib/colibri/shaders/"
              ''}

              # 2. coli dispatches relative to its own dir, so it must sit beside
              #    the engines; that also frees $out/bin/coli for the wrapper. The
              #    engine is re-exposed as $out/bin/colibri for `nix run .#engine`.
              mv $out/bin/coli $out/lib/colibri/coli
              ln -s ../lib/colibri/colibri $out/bin/colibri

              # Wrap coli through pythonEnv. COLI_ENGINE is deliberately NOT set:
              # c/coli's engine_for() routes EVERY model to GLM whenever
              # COLI_ENGINE is present, defeating per-model dispatch. Left unset,
              # coli resolves each engine beside itself (colibri / olmoe /
              # deepseek_v4 under $out/lib/colibri). PYTHONPATH makes `import
              # openai_server` / `resource_plan` / `doctor` / `v4_dsml` resolve.
              # The Vulkan build also needs the loader on LD_LIBRARY_PATH: the
              # engines dlopen libvulkan.so.1 and inherit it from this wrapper.
              makeWrapper ${pythonEnv}/bin/python $out/bin/coli \
                --add-flags "$out/lib/colibri/coli" \
                --set PYTHONPATH "$out/lib/colibri:${pythonEnv}/${pkgs.python3.sitePackages}" \
                ${pkgs.lib.optionalString vulkan "--prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath [pkgs.vulkan-loader]}"}
              runHook postInstall
            '';

            # `make test-c` isn't hermetic in a sandbox (test_ssd_probe timing,
            # Linux test_uring/io_uring); installCheckPhase validates instead.
            doCheck = false;

            # Offline verification that the multi-engine layout is correct: the
            # engines reached $out, the backfilled convert data asset is present,
            # the wrapper starts (`coli --version` argparse-exits before any model
            # load), and the serve import surface (openai_server -> v4_dsml)
            # resolves. Not versionCheckHook: coli's version string is
            # unrelated to this derivation's `version`.
            doInstallCheck = true;
            installCheckPhase = ''
              runHook preInstallCheck
              # An install check must not mutate $out; block the .pyc these imports write.
              export PYTHONDONTWRITEBYTECODE=1
              for engine in colibri glm53 olmoe qwen36 qwen38 deepseek_v41 mimo colibri-tui; do
                test -x "$out/lib/colibri/$engine"
              done
              test -f $out/lib/colibri/tools/iq3xxs_grid.json
              ${pkgs.lib.optionalString vulkan ''
                test -f $out/lib/colibri/shaders/qmatmul.spv
              ''}
              $out/bin/coli --version
              PYTHONPATH=$out/lib/colibri ${pythonEnv}/bin/python -c 'import openai_server, family_registry'
              runHook postInstallCheck
            '';

            meta = with pkgs.lib; {
              description = "Run large MoE and dense models in pure C, experts streamed from disk";
              homepage = "https://github.com/JustVugg/colibri";
              license = licenses.asl20;
              platforms = with platforms; linux ++ darwin;
              mainProgram = "coli";
            };
          };

        colibri = mkColibri {};
        colibriVulkan = mkColibri {vulkan = true;};
        isLinux = pkgs.stdenv.hostPlatform.isLinux;
      in {
        packages =
          {
            default = colibri;
            inherit colibri;
          }
          // pkgs.lib.optionalAttrs isLinux {
            # every engine built with VK=1 (any Vulkan 1.2 driver); run it with
            # COLI_VULKAN=1. CUDA is not packaged: it is unfree and needs a driver match.
            vulkan = colibriVulkan;
          };

        apps =
          {
            default = {
              type = "app";
              program = pkgs.lib.getExe colibri;
            };
            # `nix run .#engine` runs the engine binary directly, skipping the
            # coli launcher. Named "engine", not "colibri", so it doesn't shadow
            # packages.colibri (whose mainProgram is coli).
            engine = {
              type = "app";
              program = "${colibri}/bin/colibri";
            };
            # the terminal UI (a client of `coli serve`)
            tui = {
              type = "app";
              program = "${colibri}/lib/colibri/colibri-tui";
            };
          }
          // pkgs.lib.optionalAttrs isLinux {
            vulkan = {
              type = "app";
              program = "${colibriVulkan}/bin/coli";
            };
          };

        checks = {
          # builds the package and runs its installCheckPhase
          inherit colibri;
          # the model registry's Python tests: pure Python, no model, no network.
          # (`make check` builds every engine and is not hermetic in a sandbox.)
          registry =
            pkgs.runCommand "colibri-registry-tests" {
              nativeBuildInputs = [pythonEnv];
            } ''
              cp -r ${self} src
              chmod -R u+w src
              cd src/c
              python -m unittest tests.test_family_registry tests.test_formats_registry
              touch $out
            '';
        };

        formatter = pkgs.alejandra;

        devShells.default = pkgs.mkShell {
          inputsFrom = [colibri];

          packages = with pkgs;
            [
              pythonEnv
              gcc
              gnumake
              clang-tools # clangd / clang-tidy for IDE support
              pkg-config
            ]
            ++ lib.optionals isLinux [
              vulkan-loader
              vulkan-headers
              vulkan-tools
              shaderc
              mesa # lavapipe: a software Vulkan device for trying VK=1 without a GPU
            ];

          shellHook =
            pkgs.lib.optionalString isLinux ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.vulkan-loader]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              # Not exported as VK_ICD_FILENAMES: that would hide a real GPU's driver.
              export COLI_LVP_ICD="$(ls ${pkgs.mesa}/share/vulkan/icd.d/lvp_icd*.json 2>/dev/null | head -1)"
            ''
            + ''
              echo "🐦 colibrì dev shell"
              echo "  gcc: $(gcc --version | head -1)"
              echo "  python: $(python3 --version)"
              echo ""
              echo "Build one engine:   make -C c qwen36   (VK=1 for Vulkan)"
              echo "                    nix skips -march=native: add ARCH=x86-64-v3 (or NIX_ENFORCE_NO_NATIVE=0) for speed"
              echo "Build and test all: make -C c check"
              echo "Run:                python3 c/coli chat --model /path/to/model"
            ''
            + pkgs.lib.optionalString isLinux ''
              echo "Software Vulkan:    VK_ICD_FILENAMES=\$COLI_LVP_ICD COLI_VULKAN=1 (needs a VK=1 build)"
            '';
        };
      }
    );
}
