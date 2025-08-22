import git
def test_git_operations():
    repo = git.Repo('.')
    repo.git.add(all=True)
    repo.index.commit("Test commit")
    assert len(list(repo.iter_commits())) > 0
