# How to place this package on GitHub

1. Download and **extract** `Giorno_Tsallis_HMM6_GitHub_Ready_2026_10_29.zip` on your computer. Do **not** upload the ZIP expecting GitHub's repository browser to unpack it.
2. Create a new GitHub repository (suggested name `Giorno-Tsallis-HMM6`) or choose a clearly named subdirectory of your existing research repository. Avoid overwriting another paper's materials.
3. For a small first-time upload, use **Add file → Upload files**, select the **contents of the extracted folder** (many browsers also allow dragging the folder contents), then commit. The files should appear at the repository root with `README.md` visible on the home page.
4. For a reliable folder upload with Git installed, run from the extracted directory:

   ```bash
   git init
   git add README.md CITATION.cff CONTRIBUTING.md LICENSE_NOTICE.md requirements.txt .gitignore .github docs paper code data examples reports tools UPLOAD_TO_GITHUB.md
   git commit -m "Add HMM-6 referee-review manuscript and reproducibility working package"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   git push -u origin main
   ```

   Replace the placeholder URL with your **actual** repository URL. GitHub authentication is required. If uploading into an existing repository, do not reinitialize; simply copy these files into the intended location and commit normally.
5. Run `python tools/verify_release.py` before uploading. Review the outstanding caveats in `docs/VALIDATION_STATUS.md` and check the paper's current review status.
6. **Do not add** private market-price inputs, confidential code, Windows credentials or the raw 76,000 local model outputs unless you explicitly intend to distribute them and have the needed rights.
7. Before broader reuse, choose a software/data license explicitly; the included `LICENSE_NOTICE.md` deliberately grants no open-source license by default.

This package does **not** automatically upload, create a remote repository, or publish any files.
