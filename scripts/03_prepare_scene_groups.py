"""Prepare a REVIEW-ONLY draft of visual scene/leakage groups.

Designed for Colab after 01_prepare_manifest.py and 02_make_contact_sheets.py.
No new packages are needed beyond Pillow, already used by the earlier scripts.

IMPORTANT:
- Group relationships are proposed from visual inspection, not known provenance.
- The broad kitchen family contains uncertain links, merged conservatively.
- This script NEVER changes manifest.csv and NEVER creates a dev/eval split.
- Review the draft before freezing any split or selecting algorithm parameters.

Developed with Arena.ai assistance. Review and disclose this assistance in your
submission. The proposed grouping must not be described as independently
verified capture-session or physical-egg identity labels.
"""

from pathlib import Path
import argparse
import base64
import csv
import hashlib
import html
from io import BytesIO, StringIO
import json

from PIL import Image

EXPECTED_DATASET_FINGERPRINT = 'cdb04474e1663283b6848cb1efea5a248b34f505cd1d98ab85898075ed07521c'

GROUPS = [{'group_id': 'G01',
  'name': 'Gray floor and handheld views',
  'evidence': 'I001, I002 and I021 have a similar handheld egg / gray-floor context. I023 and I039 '
              'are a clearly related gray-floor pair, merged into this family conservatively.',
  'review_priority': 'Check the floor-to-handheld link; it is a precaution, not a verified '
                     'capture-session identity.',
  'subgroups': [{'name': 'Handheld gray-floor views',
                 'image_ids': ['I001', 'I002', 'I021'],
                 'note': 'Similar handheld context.'},
                {'name': 'Egg on scratched gray floor',
                 'image_ids': ['I023', 'I039'],
                 'note': 'Matching floor marks and arrangement.'}]},
 {'group_id': 'G02',
  'name': 'Striped cloth and greenery tray',
  'evidence': 'I003-I006 share the distinctive blue/yellow striped cloth. I005-I006 show a '
              'greenery tray; cloth/tray cues also link the vegetation close-ups. Keep this '
              'connected setup together.',
  'review_priority': 'Check that all greenery and cloth views are represented; the merge is '
                     'intentionally conservative.',
  'subgroups': [{'name': 'Striped cloth',
                 'image_ids': ['I003', 'I004', 'I005', 'I006'],
                 'note': 'Distinctive blue/yellow stripes; the tray is visible in the wider '
                         'views.'},
                {'name': 'Greenery / tray',
                 'image_ids': ['I007', 'I008', 'I009', 'I010', 'I040'],
                 'note': 'Repeated greenery and tray context, with cloth visible in wider '
                         'views.'}]},
 {'group_id': 'G03',
  'name': 'Ash and charcoal surface',
  'evidence': 'Repeated ash / charcoal background and egg positioning within that material.',
  'review_priority': 'Check for any additional views of this ash/charcoal setup in another group.',
  'subgroups': [{'name': 'Ash / charcoal',
                 'image_ids': ['I011', 'I022', 'I031', 'I035', 'I041', 'I045'],
                 'note': 'Keep the clean and ash-covered views together.'}]},
 {'group_id': 'G04',
  'name': 'Multicolored patterned fabric',
  'evidence': 'Matching purple, yellow, red and green patterned fabric across different camera '
              'angles.',
  'review_priority': 'Check fabric-pattern matches rather than egg appearance.',
  'subgroups': [{'name': 'Patterned fabric',
                 'image_ids': ['I012', 'I032', 'I043', 'I046'],
                 'note': 'Same distinctive fabric design.'}]},
 {'group_id': 'G05',
  'name': 'Geometric mat with earphones',
  'evidence': 'Matching blue/brown geometric mat and white earphones.',
  'review_priority': 'Check that the earphone/mat views stay together.',
  'subgroups': [{'name': 'Mat / earphones',
                 'image_ids': ['I013', 'I042', 'I044'],
                 'note': 'Matching distinctive background and prop.'}]},
 {'group_id': 'G06',
  'name': 'Dell laptop keyboard',
  'evidence': 'Same distinctive Dell laptop keyboard / palm-rest context; close and wide angles '
              'remain together.',
  'review_priority': 'Check I037, which was in the original test folder, against I014-I016.',
  'subgroups': [{'name': 'Keyboard',
                 'image_ids': ['I014', 'I015', 'I016', 'I037'],
                 'note': 'Repeated keyboard keys, branding and laptop context.'}]},
 {'group_id': 'G07',
  'name': 'Backlit hand and doorway views',
  'evidence': 'I017 and I020 clearly share the exposed-bulb wall setup. I038 is another strongly '
              'backlit hand/doorway view and is merged as a precaution.',
  'review_priority': 'The I038 doorway-to-bulb link is uncertain; conservatively keeping it on the '
                     'same side is safer than treating it as an independent scene.',
  'subgroups': [{'name': 'Exposed bulb',
                 'image_ids': ['I017', 'I020'],
                 'note': 'Matching bulb/wall context.'},
                {'name': 'Backlit doorway close-up',
                 'image_ids': ['I038'],
                 'note': 'Possible related capture context; conservatively merged.'}]},
 {'group_id': 'G08',
  'name': 'Pink CHERRY tray',
  'evidence': 'Repeated pink tray / container and CHERRY label; close-ups and the plastic-bag view '
              'are kept together.',
  'review_priority': 'Check I034, where a bag partially covers the same tray.',
  'subgroups': [{'name': 'Pink tray',
                 'image_ids': ['I018', 'I019', 'I029', 'I030', 'I033', 'I034'],
                 'note': 'Shared tray and CHERRY label where visible.'}]},
 {'group_id': 'G09',
  'name': 'White cardboard packaging',
  'evidence': 'Matching white cardboard / printed-code packaging and dark wooden background.',
  'review_priority': 'Check the packaging cues, not just the white background.',
  'subgroups': [{'name': 'White packaging',
                 'image_ids': ['I025', 'I027'],
                 'note': 'Matching cardboard scene from different distances.'}]},
 {'group_id': 'G10',
  'name': 'Gray stone-slab close-ups',
  'evidence': 'Similar mottled gray stone / slab surface in I028 and I092, conservatively grouped.',
  'review_priority': 'Check this provisional surface match; distinguish it from the scratched '
                     'floor in G01.',
  'subgroups': [{'name': 'Gray stone slab',
                 'image_ids': ['I028', 'I092'],
                 'note': 'Provisional background link; do not claim a verified capture session.'}]},
 {'group_id': 'G11',
  'name': 'Kitchen context - conservative supergroup',
  'evidence': 'A distinctive beige veined countertop, arc-patterned tiles, steel vessel and '
              'blue-printed bag link multiple kitchen subscenes. Fridge-rack, grain-bowl and tight '
              'steel-vessel views lack full context, so they are included conservatively rather '
              'than treated as independent captures.',
  'review_priority': 'This is a broad leakage-prevention family, NOT a claim that all 58 images '
                     'are one verified scene. All known multi-egg photos are concentrated here; '
                     'holding this whole family out would exceed a 22-image target. Record the '
                     'resulting held-out scene/instance-count coverage limitation.',
  'subgroups': [{'name': 'Beige countertop / multiple eggs',
                 'image_ids': ['I024',
                               'I047',
                               'I048',
                               'I050',
                               'I052',
                               'I062',
                               'I068',
                               'I079',
                               'I080',
                               'I081',
                               'I083',
                               'I086',
                               'I088',
                               'I090',
                               'I094'],
                 'note': 'Matching veined countertop and neighboring stove edge.'},
                {'name': 'Arc-patterned tile / hand / colored lids',
                 'image_ids': ['I026', 'I049', 'I095', 'I096', 'I097', 'I098', 'I102', 'I109'],
                 'note': 'Distinctive tile pattern connects lid and handheld views to the kitchen '
                         'context.'},
                {'name': 'White appliance and blue-printed bag',
                 'image_ids': ['I051', 'I053', 'I056', 'I058', 'I059', 'I070', 'I072', 'I089'],
                 'note': 'Same printed bag; tiled kitchen visible in related views.'},
                {'name': 'Orange floral container and wall',
                 'image_ids': ['I055', 'I057', 'I067', 'I074', 'I077', 'I085', 'I087'],
                 'note': 'Repeated orange floral container and adjacent wall.'},
                {'name': 'Steel water / boiling pot',
                 'image_ids': ['I060', 'I061', 'I071', 'I075', 'I076', 'I082', 'I091', 'I093'],
                 'note': 'Repeated pot, stove and tiled background; includes both Unlabeled-folder '
                         'images.'},
                {'name': 'Silver foil packets on countertop',
                 'image_ids': ['I065', 'I066', 'I078', 'I084'],
                 'note': 'Repeated foil packets on the same distinctive countertop.'},
                {'name': 'White refrigerator egg rack',
                 'image_ids': ['I036', 'I063', 'I064'],
                 'note': 'Clear within-series match; wider kitchen-session relation is uncertain, '
                         'so merge conservatively.'},
                {'name': 'Egg on white grains',
                 'image_ids': ['I054', 'I069', 'I073'],
                 'note': 'Clear within-series match; possible related kitchen capture, merged '
                         'conservatively.'},
                {'name': 'Tight single-egg steel-vessel views',
                 'image_ids': ['I099', 'I105'],
                 'note': 'The two vessel views are not proven to be the same utensil. Their '
                         'relation to the broader kitchen set is also uncertain; merge as a '
                         'precaution, not as a verified match.'}]},
 {'group_id': 'G12',
  'name': 'Green red black woven cloth',
  'evidence': 'Matching green/red/black woven fabric with repeated line pattern.',
  'review_priority': 'Check I100, I108 and I110 together.',
  'subgroups': [{'name': 'Woven cloth',
                 'image_ids': ['I100', 'I108', 'I110'],
                 'note': 'Matching distinctive textile.'}]},
 {'group_id': 'G13',
  'name': 'Handheld egg against plain light wall',
  'evidence': 'Similar plain light wall / hand context in I101 and I104; merged conservatively.',
  'review_priority': 'Plain walls provide weak provenance cues. Keep together, and do not claim '
                     'independence from all other handheld scenes.',
  'subgroups': [{'name': 'Plain-wall handheld views',
                 'image_ids': ['I101', 'I104'],
                 'note': 'Provisional context family, not verified person/egg identity.'}]},
 {'group_id': 'G14',
  'name': 'Handheld egg by green molded door',
  'evidence': 'Matching pale green molded door/panel design behind the hand.',
  'review_priority': 'Check I103, I106 and I107 together, despite original train/test folders.',
  'subgroups': [{'name': 'Green molded door',
                 'image_ids': ['I103', 'I106', 'I107'],
                 'note': 'Distinctive repeating door/panel details.'}]}]


POLICY_VERSION = "visual_scene_family_v1_DRAFT"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def dataset_fingerprint(rows):
    content = [
        {key: row[key] for key in ("image_id", "relative_path", "sha256")}
        for row in sorted(rows, key=lambda item: item["image_id"])
    ]
    return sha256_bytes(json.dumps(content, sort_keys=True, separators=(",", ":")).encode())


def csv_bytes(rows, fieldnames):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def thumbnail_uri(path):
    with Image.open(path) as original:
        image = original.convert("RGB")
        image.thumbnail((320, 320), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=87, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_review_html(rows, raw):
    esc = html.escape
    by_id = {row["image_id"]: row for row in rows}
    summary, sections = [], []
    for group in GROUPS:
        count = sum(len(part["image_ids"]) for part in group["subgroups"])
        gid = group["group_id"]
        summary.append(
            f'<tr><td><a href="#{gid}">{gid}</a></td><td>{count}</td>'
            f'<td>{esc(group["name"])}</td></tr>'
        )
        parts = []
        for part in group["subgroups"]:
            cards = []
            for image_id in part["image_ids"]:
                row = by_id[image_id]
                cards.append(
                    '<figure>'
                    f'<img loading="lazy" src="{thumbnail_uri(raw / row["relative_path"])}" '
                    f'alt="{esc(image_id)}">'
                    f'<figcaption><strong>{esc(image_id)}</strong>'
                    f' <span class="source">original {esc(row["source_split"])}/'
                    f'{esc(row.get("source_class", ""))}</span>'
                    f'<small>{esc(Path(row["relative_path"]).name.split("_jpg")[0])}</small>'
                    '</figcaption></figure>'
                )
            parts.append(
                f'<h3>{esc(part["name"])}</h3><p class="subnote">{esc(part["note"])}</p>'
                '<div class="grid">' + "".join(cards) + '</div>'
            )
        sections.append(
            f'<section id="{gid}"><h2>{gid} <span>{esc(group["name"])}</span>'
            f'<b class="count">{count} images</b></h2>'
            f'<p><strong>Reason:</strong> {esc(group["evidence"])}</p>'
            f'<p class="review"><strong>Review:</strong> {esc(group["review_priority"])}</p>'
            + "".join(parts) + '<a class="back" href="#top">Back to group list</a></section>'
        )

    css = """
    :root { color-scheme: light; --ink:#17253b; --muted:#56647a; --line:#dce3ed; }
    * { box-sizing:border-box; } body { margin:0; background:#f3f6fb; color:var(--ink);
      font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }
    main { max-width:1220px; margin:auto; padding:28px 24px 60px; }
    header { background:#142a44; color:#fff; border-radius:16px; padding:28px; }
    h1 { line-height:1.2; font-size:30px; margin:10px 0 14px; }
    .badge { display:inline-block; background:#ffda8a; color:#503700; padding:4px 10px;
      border-radius:7px; font-size:12px; font-weight:800; letter-spacing:.06em; }
    .lede { margin:8px 0; max-width:1000px; } .stats { display:flex; flex-wrap:wrap;
      gap:14px; margin-top:20px; font-weight:700; }
    .stats span { background:#243d59; border-radius:8px; padding:8px 12px; }
    .panel, section { background:white; border:1px solid var(--line); padding:22px;
      border-radius:12px; margin:22px 0; scroll-margin-top:15px; }
    .warning { border-left:5px solid #c28a12; background:#fff9e8; padding:14px 18px; }
    h2 { font-size:20px; line-height:1.35; display:flex; align-items:center; gap:12px;
      flex-wrap:wrap; margin:0 0 14px; } h2 span { flex:1; min-width:220px; }
    .count { font-size:13px; border-radius:20px; padding:5px 10px;
      background:#e8eff9; white-space:nowrap; }
    h3 { font-size:16px; margin:24px 0 6px; } p { margin:8px 0; }
    .review { background:#fff7df; border-radius:7px; padding:9px 12px; }
    .subnote, .source, small { color:var(--muted); }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
      gap:12px; margin:10px 0; } figure { margin:0; border:1px solid var(--line);
      border-radius:9px; overflow:hidden; background:#f8faff; max-width:360px; }
    img { width:100%; aspect-ratio:1; object-fit:contain; display:block; background:#edf1f5; }
    figcaption { padding:8px 10px; font-size:14px; } small { display:block; font-size:12px; }
    .source { font-size:11px; } table { width:100%; border-collapse:collapse; font-size:14px; }
    th,td { padding:9px 10px; text-align:left; border-bottom:1px solid var(--line); }
    th { background:#edf2f9; } a { color:#1858ab; } .back { display:inline-block;
      margin-top:12px; font-size:13px; } footer { color:var(--muted); font-size:13px; }
    @media(max-width:560px) { main{padding:12px;} header,.panel,section{padding:16px;}
      .grid{grid-template-columns:repeat(2,minmax(0,1fr));} h1{font-size:24px;} }
    @media print { body{background:white;} main{max-width:none;padding:0;}
      section{break-inside:auto;} figure{break-inside:avoid;} .back{display:none;} }
    """
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Egg CV | Draft scene-group review</title><style>' + css + '</style>'
        '<main id="top"><header><span class="badge">DRAFT - HUMAN CONFIRMATION PENDING</span>'
        '<h1>Egg CV: scene-group review</h1>'
        '<p class="lede">Same identifiable background, props or connected capture context '
        'should stay on one side of the future dev/eval split. These groups are based '
        'on visual inspection, not verified source/session metadata.</p>'
        '<div class="stats"><span>110 originals</span><span>14 proposed groups</span>'
        '<span>No split created</span><span>AI-assisted draft</span></div></header>'
        '<div class="panel"><h2>What to check / kya dekhna hai</h2>'
        '<ol><li>Scroll through the groups. Same setup ki photo kisi aur group mein '
        'dikh rahi ho, toh dono image IDs note karo.</li>'
        '<li>G01, G07, G10 and G13 have provisional links. G11 deliberately merges '
        'some uncertain kitchen close-ups to avoid optimistic independence assumptions.</li>'
        '<li>Confirm or correct scene relationships BEFORE split selection and algorithm '
        'tuning. Never change groups in response to evaluation scores.</li></ol>'
        '<p class="warning"><strong>Important limitation:</strong> G11 contains 58 images, '
        'including the clearly visible multiple-egg scenes. A whole-group split targeting '
        '22 evaluation images cannot put G11 in evaluation. Held-out instance-count and '
        'scene diversity will therefore be limited; report this honestly. Do not split '
        'related photos merely to improve the ratio or coverage.</p>'
        '<p>No provenance is available to guarantee that the same physical egg never '
        'appears in different groups. This is a practical leakage-reduction policy, '
        'not proof of independent people, eggs or capture sessions. It is not based '
        'on segmentation scores or perceived algorithm difficulty.</p></div>'
        '<div class="panel"><h2>Proposed groups</h2><table><thead><tr><th>ID</th>'
        '<th>Images</th><th>Context</th></tr></thead><tbody>' + "".join(summary) +
        '</tbody></table></div>' + "".join(sections) +
        '<footer>Original train/valid/test labels are source metadata only, not the new '
        'assignment split. These previews do not create masks, augment data, train a '
        'model or evaluate a segmentation method. Draft prepared with Arena.ai assistance; '
        'human review and disclosure are required.</footer></main></html>'
    )


def main(project_root):
    project = Path(project_root)
    metadata = project / "metadata"
    raw = project / "data" / "raw"
    manifest_path = metadata / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Manifest not found. Mount Drive and run Step 01 first.")
    if (metadata / "split.csv").exists():
        raise RuntimeError("A split already exists. Do not change grouping after a split freeze; share this error.")
    manifest_before = manifest_path.read_bytes()
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"image_id", "relative_path", "source_split", "sha256"}
        if not required.issubset(fields):
            raise ValueError("Required manifest columns are missing. Share this error.")
        rows = sorted(list(reader), key=lambda row: row["image_id"])

    expected_ids = {f"I{number:03}" for number in range(1, 111)}
    if len(rows) != 110 or {row["image_id"] for row in rows} != expected_ids:
        raise ValueError("This grouping expects the same 110 image IDs. Share your manifest before continuing.")
    fingerprint = dataset_fingerprint(rows)
    if fingerprint != EXPECTED_DATASET_FINGERPRINT:
        raise ValueError(
            "Your manifest differs from the inspected dataset. No grouping was applied. "
            "Share metadata/manifest.csv so the ID-to-photo mapping can be checked."
        )

    assignments = {}
    for group in GROUPS:
        for part in group["subgroups"]:
            for image_id in part["image_ids"]:
                if image_id in assignments:
                    raise ValueError(f"Duplicate proposed assignment: {image_id}")
                assignments[image_id] = (group, part)
    if set(assignments) != expected_ids:
        raise ValueError("Draft grouping does not cover all IDs exactly once.")

    raw_root = raw.resolve()
    for row in rows:
        path = (raw / row["relative_path"]).resolve()
        if not path.is_relative_to(raw_root) or not path.is_file():
            raise ValueError(f"Missing or unsafe image path for {row['image_id']}")
        if sha256_bytes(path.read_bytes()) != row["sha256"]:
            raise ValueError(f"Raw image changed: {row['image_id']}. Stop and share this error.")
        with Image.open(path) as image:
            image.verify()
        existing = row.get("scene_group_id", "").strip()
        proposed = assignments[row["image_id"]][0]["group_id"]
        if existing and existing != proposed:
            raise ValueError(f"An existing annotation differs for {row['image_id']}; it was not overwritten.")

    new_fields = fields.copy()
    for field in ("scene_group_id", "scene_group_name", "scene_subgroup", "grouping_status", "review_notes"):
        if field not in new_fields:
            new_fields.append(field)
    draft_rows = []
    for original in rows:
        row = original.copy()
        group, part = assignments[row["image_id"]]
        row.update({
            "scene_group_id": group["group_id"],
            "scene_group_name": group["name"],
            "scene_subgroup": part["name"],
            "grouping_status": "DRAFT_AI_ASSISTED_REQUIRES_REVIEW",
        })
        row.setdefault("review_notes", "")
        draft_rows.append(row)

    summary = []
    for group in GROUPS:
        summary.append({
            "scene_group_id": group["group_id"],
            "image_count": sum(len(part["image_ids"]) for part in group["subgroups"]),
            "name": group["name"],
            "evidence": group["evidence"],
            "review_priority": group["review_priority"],
            "status": "DRAFT_REQUIRES_REVIEW",
        })

    policy = {
        "policy_version": POLICY_VERSION,
        "status": "DRAFT_REQUIRES_HUMAN_CONFIRMATION",
        "dataset_fingerprint": fingerprint,
        "input_manifest_sha256": sha256_bytes(manifest_before),
        "rule": "Keep connected visual capture contexts together; conservatively merge uncertain related views. Use scene_group_id, not scene_subgroup, for the future split.",
        "review_source": "Visual inspection of all six contact sheets and selected original images from the inspected alternative dataset ZIP; raw hashes checked against the user's manifest.",
        "limitations": [
            "No verified session, person or physical-egg identities; cross-group physical-egg reuse cannot be ruled out.",
            "Some connections are provisional, especially G01, G07, G10, G13 and the broader kitchen family.",
            "G11 has 58 images and all clearly visible multiple-egg scenes in the reviewed contact sheets. A 22-image whole-group eval cannot include G11; report limited held-out instance-count/scene coverage.",
            "No split, reference mask, classical prediction, experiment or metric is produced by this script.",
        ],
        "ai_assistance": "Draft grouping and code prepared with Arena.ai assistance; human review pending; disclose assistance in the final report.",
        "groups": GROUPS,
    }

    outputs = {
        metadata / "scene_groups_draft.csv": csv_bytes(draft_rows, new_fields),
        metadata / "scene_group_summary_draft.csv": csv_bytes(summary, list(summary[0])),
        metadata / "scene_group_policy_draft.json": (json.dumps(policy, indent=2) + "\n").encode("utf-8"),
        metadata / "scene_group_review_draft.html": build_review_html(rows, raw).encode("utf-8"),
    }
    # Check ALL output conflicts before writing any file. Preserve manual edits.
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError(f"Existing draft differs and was preserved: {path}. Share this message; do not delete your edits.")
    for path, content in outputs.items():
        if not path.exists():
            path.write_bytes(content)

    if manifest_path.read_bytes() != manifest_before:
        raise RuntimeError("Unexpected manifest change detected. Stop and share this error.")

    print("Images verified: 110")
    print(f"Draft groups: {len(GROUPS)}")
    print("Draft grouping coverage: 110/110")
    print("Missing or duplicate assignments: 0")
    print("Original manifest unchanged: YES")
    print("Dev/eval split created: NO")
    print("Review status: DRAFT - awaiting your confirmation")
    print("\nGROUP SUMMARY")
    for item in summary:
        print(f"{item['scene_group_id']} | {item['image_count']:2} images | {item['name']}")
    print("\nFiles saved:")
    for path in outputs:
        print(path)
    print("\nReview the grouped gallery, then confirm or report image-ID corrections.")
    print("Use scene_group_id (whole family), NOT scene_subgroup, for the future split.")
    return list(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/content/drive/MyDrive/Egg_CV_Task")
    args = parser.parse_args()
    main(args.project_root)
