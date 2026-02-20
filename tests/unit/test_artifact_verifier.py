"""
ARTIFACT VERIFIER TESTS

Tests for artifact verification logic.
Verifies inline vs file detection and verification rules.
"""
import pytest
import sys
import os
import tempfile
import json

# Add app directory to path
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)


class TestArtifactVerifierInit:
    """Test artifact verifier initialization."""

    def test_verifier_imports(self):
        """Artifact verifier must import."""
        from artifact_verifier import ArtifactVerifier, artifact_verifier
        assert ArtifactVerifier is not None
        assert artifact_verifier is not None

    def test_verifier_creates_base_path(self):
        """Verifier must create base path if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from artifact_verifier import ArtifactVerifier
            base_path = os.path.join(tmpdir, "artifacts")
            verifier = ArtifactVerifier(base_path=base_path)
            assert os.path.exists(base_path)


class TestInlineContentDetection:
    """Test inline vs file path detection."""

    def test_is_inline_content_json(self):
        """JSON-like content must be detected as inline."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        
        # JSON object
        assert verifier._is_inline_content('{"key": "value"}', "file") == True
        
        # JSON array
        assert verifier._is_inline_content('[1, 2, 3]', "file") == True

    def test_is_inline_content_multiline(self):
        """Multiline content must be detected as inline."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        
        # Content with newlines
        assert verifier._is_inline_content("line1\nline2\nline3", "file") == True

    def test_is_inline_content_long_text(self):
        """Long text without file-like patterns must be inline."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        
        # Long text (>200 chars)
        long_text = "x" * 250
        assert verifier._is_inline_content(long_text, "file") == True

    def test_is_file_path_with_extension(self):
        """Path with extension must NOT be inline."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        
        assert verifier._is_inline_content("report.md", "file") == False
        assert verifier._is_inline_content("data.json", "file") == False
        assert verifier._is_inline_content("output.txt", "file") == False

    def test_is_file_path_with_slash(self):
        """Path with slash must NOT be inline."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        
        assert verifier._is_inline_content("path/to/file.md", "file") == False
        assert verifier._is_inline_content("/absolute/path", "file") == False

    def test_is_inline_content_non_file_kind(self):
        """Non-file content_kind must not trigger inline detection."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        
        # Even if looks like inline, if content_kind is not "file", return False
        assert verifier._is_inline_content('{"key": "value"}', "db") == False
        assert verifier._is_inline_content('{"key": "value"}', "vector") == False


class TestVerifyInlineContent:
    """Test inline content verification."""

    def test_verify_inline_json(self):
        """Valid JSON inline content must pass."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        content = json.dumps({"result": "success", "data": [1, 2, 3]})
        
        results = verifier._verify_inline_content("FILE", content)
        
        # Check results
        result_dict = {r.name: r for r in results}
        
        assert result_dict.get("inline_content_exists").passed == True
        assert result_dict.get("json_valid").passed == True

    def test_verify_inline_text(self):
        """Valid text inline content must pass."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        content = "This is a valid inline text content for testing."
        
        results = verifier._verify_inline_content("FILE", content)
        
        result_dict = {r.name: r for r in results}
        
        assert result_dict.get("inline_content_exists").passed == True
        assert result_dict.get("content_min_length").passed == True

    def test_verify_inline_empty_fails(self):
        """Empty content must fail."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        
        results = verifier._verify_inline_content("FILE", "")
        
        result_dict = {r.name: r for r in results}
        assert result_dict.get("inline_content_exists").passed == False


class TestArtifactTypeValidation:
    """Test artifact type validation."""

    def test_valid_types(self):
        """Valid artifact types must pass."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        
        valid_types = ["FILE", "KNOWLEDGE", "DATASET", "REPORT", "LINK", "EXECUTION_LOG"]
        
        for artifact_type in valid_types:
            result = verifier._verify_type(artifact_type)
            assert result.passed == True

    def test_invalid_type_fails(self):
        """Invalid artifact type must fail."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        
        result = verifier._verify_type("INVALID_TYPE")
        assert result.passed == False


class TestOverallStatus:
    """Test overall verification status determination."""

    def test_all_passed_returns_passed(self):
        """All checks passed must return 'passed'."""
        from artifact_verifier import ArtifactVerifier, VerificationResult
        
        verifier = ArtifactVerifier()
        
        results = [
            VerificationResult("check1", True, "ok"),
            VerificationResult("check2", True, "ok"),
        ]
        
        status = verifier.get_overall_status(results)
        assert status == "passed"

    def test_critical_failed_returns_failed(self):
        """Critical check failed must return 'failed'."""
        from artifact_verifier import ArtifactVerifier, VerificationResult
        
        verifier = ArtifactVerifier()
        
        results = [
            VerificationResult("file_exists", False, "not found"),
            VerificationResult("check2", True, "ok"),
        ]
        
        status = verifier.get_overall_status(results)
        assert status == "failed"

    def test_empty_results_returns_failed(self):
        """Empty results must return 'failed'."""
        from artifact_verifier import ArtifactVerifier
        
        verifier = ArtifactVerifier()
        
        status = verifier.get_overall_status([])
        assert status == "failed"
