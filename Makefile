test:
	python -m unittest discover --start-directory tests

docupdate:
	rm -rf docs/source/New_MASCADO.*
	cd docs; \
	sphinx-apidoc -MeT -o source/ ../New_MASCADO

docbuild:
	cd docs; \
	make html

docshow:
	xdg-open docs/build/html/py-modindex.html
