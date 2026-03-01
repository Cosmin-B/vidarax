use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const ZVEC_GIT_REF: &str = "v0.2.0";

/// Expected SHA-1 commit hash for the `v0.2.0` tag of alibaba/zvec.
///
/// Obtained from `git rev-parse HEAD` after the first successful clone.
/// Update this constant whenever `ZVEC_GIT_REF` is bumped to a new release.
const ZVEC_EXPECTED_COMMIT: &str = "385a030284a1c0a61c132ccefc63171439242bc9";

fn ensure_zvec_source(workspace_dir: &Path) -> PathBuf {
    let zvec_src = workspace_dir.join("vendor/zvec");

    if zvec_src.join("CMakeLists.txt").exists() {
        println!("cargo:warning=zvec source already present");
        verify_commit_hash(&zvec_src);
        return zvec_src;
    }

    println!(
        "cargo:warning=Cloning zvec {} (this may take a few minutes)...",
        ZVEC_GIT_REF
    );
    let _ = std::fs::create_dir_all(zvec_src.parent().unwrap());

    // Clone into a staging directory first; rename only after hash verification
    // so a partial clone or a tampered tag never leaves a broken vendor tree.
    let staging = zvec_src.with_extension("_staging");
    let _ = std::fs::remove_dir_all(&staging);

    let status = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "--branch",
            ZVEC_GIT_REF,
            "--recursive",
            "https://github.com/alibaba/zvec.git",
            staging.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute git clone. Please ensure git is installed.");

    if !status.success() {
        panic!("git clone failed. Please check your network connection and that git is installed.");
    }

    verify_commit_hash(&staging);

    std::fs::rename(&staging, &zvec_src)
        .expect("Failed to rename staging clone to vendor/zvec");

    zvec_src
}

/// Verify the HEAD commit of a cloned repository against [`ZVEC_EXPECTED_COMMIT`].
///
/// Panics if the hash cannot be read or does not match, preventing a
/// supply-chain attack where the remote tag is force-pushed to a different
/// commit.
///
/// If [`ZVEC_EXPECTED_COMMIT`] still contains the placeholder value this
/// function prints the actual hash as a cargo warning so the developer can
/// fill it in, then returns without panicking (bootstrap mode).
fn verify_commit_hash(repo_dir: &Path) {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_dir)
        .output()
        .expect("Failed to run `git rev-parse HEAD` in cloned zvec directory");

    if !output.status.success() {
        panic!(
            "git rev-parse HEAD failed in {}",
            repo_dir.display()
        );
    }

    let actual = String::from_utf8_lossy(&output.stdout)
        .trim()
        .to_string();

    if ZVEC_EXPECTED_COMMIT.starts_with("FILL_IN") {
        // Bootstrap: developer has not yet pinned the hash.  Emit it so they
        // can update the constant, but do not block the build.
        println!(
            "cargo:warning=zvec supply-chain: set ZVEC_EXPECTED_COMMIT = {:?} in build.rs",
            actual
        );
        return;
    }

    if actual != ZVEC_EXPECTED_COMMIT {
        panic!(
            "zvec supply-chain check failed: expected commit {}, got {}. \
             If you intentionally bumped ZVEC_GIT_REF, update ZVEC_EXPECTED_COMMIT in build.rs.",
            ZVEC_EXPECTED_COMMIT,
            actual
        );
    }

    println!("cargo:warning=zvec commit hash verified: {}", actual);
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir.parent().expect("Expected parent directory");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    println!("cargo:rerun-if-env-changed=ZVEC_GIT_REF");
    println!("cargo:rerun-if-env-changed=ZVEC_BUILD_TYPE");
    println!("cargo:rerun-if-env-changed=ZVEC_BUILD_PARALLEL");
    println!("cargo:rerun-if-env-changed=ZVEC_CPU_ARCH");
    println!("cargo:rerun-if-env-changed=ZVEC_OPENMP");

    let zvec_src = ensure_zvec_source(workspace_dir);
    let zvec_build = zvec_src.join("build");
    let zvec_lib = zvec_build.join("lib");

    let build_type = env::var("ZVEC_BUILD_TYPE").unwrap_or_else(|_| "Release".to_string());
    let parallel_jobs = env::var("ZVEC_BUILD_PARALLEL")
        .map(|s| s.parse::<usize>().unwrap_or_else(|_| num_cpus()))
        .unwrap_or_else(|_| num_cpus());

    let wrapper_dir = manifest_dir.join("zvec-c-wrapper");
    let wrapper_build = out_dir.join("zvec-c-wrapper-build");

    let zvec_built = zvec_lib.join("libzvec_db.a");
    if !zvec_built.exists() {
        println!("cargo:warning=Building zvec C++ library...");
        build_zvec(&zvec_src, &zvec_build, &build_type, parallel_jobs);
    } else {
        println!("cargo:warning=zvec C++ library already built");
    }

    let wrapper_built = wrapper_build.join("libzvec_c_wrapper.a");
    if !wrapper_built.exists() {
        println!("cargo:warning=Building C wrapper...");
        build_c_wrapper(
            &wrapper_dir,
            &wrapper_build,
            &zvec_src,
            &build_type,
            parallel_jobs,
        );
    } else {
        println!("cargo:warning=C wrapper already built");
    }

    generate_bindings(&wrapper_dir);

    link_libraries(&zvec_lib, &wrapper_build);
}

fn build_zvec(_src: &Path, build: &Path, build_type: &str, parallel_jobs: usize) {
    let _ = std::fs::create_dir_all(build);

    let mut cmake_args = vec![
        format!("-DCMAKE_BUILD_TYPE={}", build_type),
        "-DBUILD_PYTHON_BINDINGS=OFF".to_string(),
        "-DBUILD_TOOLS=OFF".to_string(),
    ];

    if let Ok(arch) = env::var("ZVEC_CPU_ARCH") {
        cmake_args.push(format!("-DENABLE_{}=ON", arch));
    }

    if env::var("ZVEC_OPENMP")
        .map(|v| v == "ON" || v == "1")
        .unwrap_or(false)
    {
        cmake_args.push("-DENABLE_OPENMP=ON".to_string());
    }

    cmake_args.push("..".to_string());

    run(
        Command::new("cmake").current_dir(build).args(&cmake_args),
        "cmake configure for zvec",
    );

    run(
        Command::new("make")
            .current_dir(build)
            .args(["-j", parallel_jobs.to_string().as_str()]),
        "make for zvec",
    );
}

fn build_c_wrapper(
    wrapper_dir: &Path,
    build: &Path,
    zvec_src: &Path,
    build_type: &str,
    parallel_jobs: usize,
) {
    let _ = std::fs::create_dir_all(build);

    run(
        Command::new("cmake").current_dir(build).args([
            format!("-DZVEC_SRC_DIR={}", zvec_src.display()).as_str(),
            format!("-DCMAKE_BUILD_TYPE={}", build_type).as_str(),
            wrapper_dir.to_str().expect("Invalid wrapper dir path"),
        ]),
        "cmake configure for C wrapper",
    );

    run(
        Command::new("make")
            .current_dir(build)
            .args(["-j", parallel_jobs.to_string().as_str()]),
        "make for C wrapper",
    );
}

fn generate_bindings(wrapper_dir: &Path) {
    let header = wrapper_dir.join("include/zvec_c.h");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    let bindings = bindgen::Builder::default()
        .header(header.to_str().expect("Invalid header path"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate_comments(true)
        .clang_arg("-I/usr/include")
        .clang_arg("-I/usr/local/include")
        .clang_arg("-I/usr/lib/gcc/aarch64-linux-gnu/13/include")
        .clang_arg("-I/usr/include/c++/13")
        .clang_arg("-I/usr/include/aarch64-linux-gnu/c++/13")
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn link_libraries(zvec_lib: &Path, wrapper_build: &Path) {
    // C wrapper
    println!("cargo:rustc-link-search=native={}", wrapper_build.display());
    println!("cargo:rustc-link-lib=static=zvec_c_wrapper");

    // zvec component libraries path
    println!("cargo:rustc-link-search=native={}", zvec_lib.display());

    // External third-party libraries (built in build/external/usr/local/lib)
    let external_lib = zvec_lib.parent().unwrap().join("external/usr/local/lib");
    println!("cargo:rustc-link-search=native={}", external_lib.display());

    // Arrow build directory (contains thrift and many other libs)
    let arrow_build = zvec_lib
        .parent()
        .unwrap()
        .join("thirdparty/arrow/arrow/src/ARROW.BUILD-build");
    println!(
        "cargo:rustc-link-search=native={}",
        arrow_build.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        arrow_build.join("release").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        arrow_build.join("re2_ep-install/lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        arrow_build.join("utf8proc_ep-install/lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        arrow_build
            .join("zlib_ep/src/zlib_ep-install/lib")
            .display()
    );

    // Boost libraries
    let boost_build = arrow_build.join("_deps/boost-build/libs");
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("atomic").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("charconv").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("chrono").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("container").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("date_time").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("locale").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        boost_build.join("thread").display()
    );

    // LZ4
    let lz4_build = zvec_lib
        .parent()
        .unwrap()
        .join("thirdparty/lz4/lz4/src/Lz4.BUILD/lib");
    println!("cargo:rustc-link-search=native={}", lz4_build.display());

    // All libraries as whole-archive to ensure they're linked in tests
    // (Cargo doesn't propagate regular static lib links to test binaries)
    // Note: zvec_core.a bundles core_knn_* libraries
    // Note: zvec_db.a bundles zvec_common, zvec_index, zvec_proto, zvec_sqlengine
    let whole_archive_libs = ["zvec_core", "zvec_ailego", "zvec_db"];
    for lib in &whole_archive_libs {
        println!("cargo:rustc-link-lib=static:+whole-archive={}", lib);
    }

    // Third-party dependencies (whole-archive for test linking)
    // Note: 'z', 'utf8proc', 're2', 'thrift' are included in arrow_bundled_dependencies
    let thirdparty_libs = [
        "parquet",
        "arrow_acero",
        "arrow_dataset",
        "arrow_compute",
        "arrow",
        "arrow_bundled_dependencies",
        "roaring",
        "rocksdb",
        "lz4",
        "protobuf",
        "protoc",
        "boost_thread",
        "boost_atomic",
        "boost_chrono",
        "boost_container",
        "boost_date_time",
        "boost_locale",
        "boost_charconv",
        "glog",
        "gflags_nothreads",
        "antlr4-runtime",
    ];
    for lib in &thirdparty_libs {
        println!("cargo:rustc-link-lib=static:+whole-archive={}", lib);
    }

    // System libraries — macOS ships libc++ (Clang), Linux ships libstdc++ (GCC)
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        // ICU libs required by boost_locale (pulled in by zvec's Arrow dependency)
        println!("cargo:rustc-link-lib=icui18n");
        println!("cargo:rustc-link-lib=icuuc");
        println!("cargo:rustc-link-lib=icudata");
    }
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");
}

fn run(cmd: &mut Command, context: &str) {
    println!("cargo:warning=Running: {:?}", cmd);
    let status = cmd.status().unwrap_or_else(|_| {
        panic!("Failed to execute command: {}", context);
    });
    if !status.success() {
        panic!("Command failed ({}): {:?}", context, cmd);
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}
