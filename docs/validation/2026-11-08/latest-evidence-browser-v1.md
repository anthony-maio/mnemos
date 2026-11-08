# Latest evidence page: local browser verification

The website draft at `125d0a94c066ae0cd9142d271596a9fdbddb70fd` passes local
desktop and mobile browser checks. Edge 141.0.3537.71 rendered the homepage and
evaluation checkpoint at 1440x1000 and 390x844. Navigation from the homepage
reached the expected checkpoint, the current evidence text was visible, the
historical quality table retained four rows, and neither viewport had page-wide
horizontal overflow. The mobile checkpoint screenshot was also inspected.

All five receipt links were clicked in each viewport. Each navigation returned
HTTP 200, valid JSON and exactly the local file bytes, including the two new
optional-workflow and discriminative-support receipts. No page errors occurred.
This verifies ten local receipt navigations, not external links or a deployment.

The first invocation used an environment without Playwright and failed before
starting a browser or server. The existing system Python Playwright installation
then ran the unchanged script successfully. Its process exited with code zero;
browser cleanup and the local HTTP server shutdown completed. The website
working tree remains clean. No runtime or configuration was activated.

The [receipt](latest-evidence-browser-v1-receipt.json) binds the source revision,
page/style hashes, downloaded receipt hashes and verified six-member private
archive containing the script, four screenshots and result. These checks do not
alter any quality, latency or pilot result. Publication remains pending.
