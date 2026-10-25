"""Run from demo-app: .venv/bin/python -m unittest discover -s tests."""
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main


class StartupTests(unittest.TestCase):
    def test_pages_start_without_document_uploads(self):
        with (
            patch.dict(os.environ, {"ENABLE_DOCUMENT_SEARCH": "false"}),
            patch.object(main, "init_vector_store", new_callable=AsyncMock) as faq,
            patch.object(main, "init_resume_store", new_callable=AsyncMock) as resume,
            TestClient(main.app) as client,
        ):
            for page in ["/", "/finance.html", "/podcaster.html", "/indigo-chat.html"]:
                with self.subTest(page=page):
                    self.assertEqual(client.get(page).status_code, 200)
            faq.assert_not_awaited()
            resume.assert_not_awaited()

    def test_document_search_requires_explicit_opt_in(self):
        with (
            patch.dict(os.environ, {"ENABLE_DOCUMENT_SEARCH": "true"}),
            patch.object(main, "init_vector_store", new_callable=AsyncMock) as faq,
            patch.object(main, "init_resume_store", new_callable=AsyncMock) as resume,
            TestClient(main.app),
        ):
            faq.assert_awaited_once_with(str(main.FAQ_PDF))
            resume.assert_awaited_once_with(str(main.RESUME_PDF))

    def test_exa_tools_construct_without_interactive_install(self):
        from app_agents.finance_agent import _build_researcher as finance_researcher
        from app_agents.podcaster_agent import _build_researcher as podcast_researcher

        self.assertTrue(finance_researcher().tools)
        self.assertTrue(podcast_researcher().tools)


if __name__ == "__main__":
    unittest.main()
