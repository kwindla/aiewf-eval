# Local filler-token report

The report is generated from:

```text
scripts/build_filler_report.py
```

The generated, self-contained page is:

```text
docs/filler-token-latent-scratchpad-study.html
```

After editing the generator, rebuild it from the repository root:

```bash
python3 scripts/build_filler_report.py
```

Serve only the report (not the repository) on localhost:

```bash
python3 scripts/serve_filler_report.py --host 127.0.0.1 --port 8765
```

The server reads the HTML file for every request, so a browser refresh shows a
new build without restarting the server. It exposes only `/`, `/index.html`, and
`/healthz`; all other paths return 404.

To make the local server remotely reachable:

```bash
ngrok http http://127.0.0.1:8765
```

The public ngrok URL is temporary unless the ngrok account is configured with a
reserved domain. Anyone with the URL can read the report.

For a restartable per-user deployment, link and enable the checked-in units:

```bash
systemctl --user link "$PWD/ops/systemd/aiewf-filler-report.service"
systemctl --user link "$PWD/ops/systemd/aiewf-filler-report-ngrok.service"
systemctl --user enable --now \
  aiewf-filler-report.service \
  aiewf-filler-report-ngrok.service
```

Check the current public URL at `http://127.0.0.1:4040/api/tunnels`, or inspect
the tunnel service log with:

```bash
journalctl --user -u aiewf-filler-report-ngrok.service -n 20 --no-pager
```
