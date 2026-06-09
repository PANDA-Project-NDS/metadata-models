from unittest.mock import patch

import golden.agents.judge as judge
import golden.main as main
from golden.agents.judge import load_coverage_sections
from golden.main import discover_journals


# --- 5b. Coverage Parsing (load_coverage_sections) ---


class TestCoverageParsing:
    def test_load_coverage_sections_parses_markdown(self, tmp_path):
        """Markdown with ## Publisher A / ## Publisher B headings is split into sections."""
        coverage_file = tmp_path / "coverage.md"
        coverage_file.write_text(
            "### Publisher A (2 journals: j1, j2)\nContent A\n\n### Publisher B (2 journals: j3, j4)\nContent B"
        )

        with patch.object(judge, "COVERAGE_PATH", coverage_file):
            result = load_coverage_sections()

        assert len(result) == 2
        assert "publisher a" in result
        assert "publisher b" in result

    def test_coverage_key_remap(self, tmp_path):
        """Heading keys like 'springer nature' are remapped to 'springer_nature'."""
        coverage_file = tmp_path / "coverage.md"
        coverage_file.write_text(
            "### Springer Nature (1 journals: j1)\nSN content"
        )

        with patch.object(judge, "COVERAGE_PATH", coverage_file):
            result = load_coverage_sections()

        assert "springer_nature" in result
        assert "springer nature" not in result


# --- 5c. Journal Discovery (discover_journals) ---


class TestJournalDiscovery:
    def test_discovers_journals_with_markdown(self, tmp_path):
        """Directories with .md files are discovered; empty dirs are skipped."""
        # Simulate journal-samples structure
        pub_dir = tmp_path / "pub1" / "extracted"
        (pub_dir / "j1").mkdir(parents=True)
        (pub_dir / "j2").mkdir()
        (pub_dir / "j3").mkdir()  # empty — should be skipped

        (pub_dir / "j1" / "editors.md").write_text("# Editors")
        (pub_dir / "j2" / "about.md").write_text("# About")

        with patch.object(main, "SAMPLES_ROOT", tmp_path):
            results = list(discover_journals())

        journals = [(j.publisher, j.journal) for j in results]
        assert ("pub1", "j1") in journals
        assert ("pub1", "j2") in journals
        assert not any(j.journal == "j3" for j in results)

    def test_publisher_filter(self, tmp_path):
        """publisher_filter limits results to matching publisher."""
        pub1 = tmp_path / "pub1" / "extracted"
        pub2 = tmp_path / "pub2" / "extracted"
        (pub1 / "j1").mkdir(parents=True)
        (pub2 / "j2").mkdir(parents=True)
        (pub1 / "j1" / "editors.md").write_text("# Editors")
        (pub2 / "j2" / "editors.md").write_text("# Editors")

        with patch.object(main, "SAMPLES_ROOT", tmp_path):
            results = list(discover_journals(publisher_filter="pub1"))

        assert len(results) == 1
        assert results[0].publisher == "pub1"

    def test_journal_filter(self, tmp_path):
        """journal_filter limits results to matching journal."""
        pub_dir = tmp_path / "pub1" / "extracted"
        (pub_dir / "j1").mkdir(parents=True)
        (pub_dir / "j2").mkdir()
        (pub_dir / "j1" / "editors.md").write_text("# Editors")
        (pub_dir / "j2" / "editors.md").write_text("# Editors")

        with patch.object(main, "SAMPLES_ROOT", tmp_path):
            results = list(discover_journals(journal_filter="j2"))

        assert len(results) == 1
        assert results[0].journal == "j2"

    def test_skips_golden_directory(self, tmp_path):
        """The 'golden' directory is excluded from discovery."""
        (tmp_path / "golden" / "j1").mkdir(parents=True)
        (tmp_path / "pub1" / "extracted" / "j1").mkdir(parents=True)
        (tmp_path / "golden" / "j1" / "editors.md").write_text("# Editors")
        (tmp_path / "pub1" / "extracted" / "j1" / "editors.md").write_text("# Editors")

        with patch.object(main, "SAMPLES_ROOT", tmp_path):
            results = list(discover_journals())

        assert all(j.publisher != "golden" for j in results)
