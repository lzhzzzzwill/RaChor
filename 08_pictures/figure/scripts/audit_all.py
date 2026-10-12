"""Audit every exported PDF and the shared plotting implementation."""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill-scripts', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.skill_scripts))
    from audit_figure_collisions import audit_pdf
    from audit_pdf_text import audit_pdf as audit_text
    from validate_figure import validate_source, summarize
    base = Path(__file__).resolve().parent.parent
    qa = Path(tempfile.mkdtemp(prefix='nature-redraw-audit-'))
    qa.mkdir(exist_ok=True)
    # Inspect the caller together with the helper that configures fonts and export.
    combined = '\n'.join((base/'scripts'/name).read_text().replace('from __future__ import annotations', '')
                         for name in ['publication_style.py', 'main.py', 'response_options.py', 'supplementary.py'])
    findings = validate_source(combined, 'python')
    source = {'summary': summarize(findings, False), 'findings': [asdict(f) for f in findings]}
    (qa/'source_preflight.json').write_text(json.dumps(source, indent=2))
    summary = []
    for p in sorted((base/'output').rglob('*.pdf')):
        if 'qa' in p.relative_to(base/'output').parts:
            continue
        c = audit_pdf(p)
        t = audit_text(p.read_bytes())
        rel = p.relative_to(base/'output')
        out = qa/rel.parent
        out.mkdir(exist_ok=True, parents=True)
        c['pdf'] = str(rel)
        (out/(p.stem+'.collision.json')).write_text(json.dumps(c, indent=2))
        (out/(p.stem+'.text.json')).write_text(json.dumps(t, indent=2))
        summary.append({'file': str(rel), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                        'collision': c['verdict'], 'fail': c['summary']['fail'],
                        'warn': c['summary']['warn'], 'font_min': t['minimum_found_pt'],
                        'text_auditable': t['auditable'], 'small_glyphs': t['below_minimum_count']})
        if c['summary']['fail'] or c['summary']['warn'] or t['below_minimum_count']:
            print(rel, c['summary'], [(f['kind'],f['text']) for f in c['findings']], flush=True)
    (qa/'summary.json').write_text(json.dumps(summary, indent=2))
    print('PDF count:', len(summary))
    print('Source preflight:', source['summary'])
    print('PDF failures:', sum(s['fail'] for s in summary), 'warnings:', sum(s['warn'] for s in summary))
    print('Minimum glyph size:', min(s['font_min'] for s in summary if s['font_min']))
    if any(s['fail'] or s['small_glyphs'] or not s['text_auditable'] for s in summary):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
