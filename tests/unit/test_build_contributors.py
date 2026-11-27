"""The docs' contributor grid: GitHub's list, plus the contributors it leaves out (co-authors of squashed commits)."""

from scripts.utilities import build_contributors as bc

CONFIG = {
    "repository": "owner/repo",
    "lead": "lead",
    "paper": ["paper1"],
    "names": {"coauthor": "Co Author"},
    "extra": ["coauthor", "author", "hidden"],
    "exclude": ["hidden", "bot[bot]"],
}
GITHUB = [{"login": "author", "html_url": "https://github.com/author"},
          {"login": "paper1"}, {"login": "lead"}, {"login": "bot[bot]", "type": "Bot"}]


def fake_get(url: str):
    if "/contributors" in url:
        return GITHUB
    return {"name": url.rsplit("/", 1)[-1].title()}


def test_extra_contributors_follow_githubs_list(monkeypatch):
    monkeypatch.setattr(bc, "_get", fake_get)
    people = bc.contributors(CONFIG)
    assert [p["login"] for p in people] == ["lead", "paper1", "author", "coauthor"]
    coauthor = people[-1]
    assert coauthor == {"login": "coauthor", "name": "Co Author", "role": "code",
                        "url": "https://github.com/coauthor"}


def test_no_extra_key_leaves_githubs_list(monkeypatch):
    monkeypatch.setattr(bc, "_get", fake_get)
    config = {key: value for key, value in CONFIG.items() if key != "extra"}
    assert [p["login"] for p in bc.contributors(config)] == ["lead", "paper1", "author"]
