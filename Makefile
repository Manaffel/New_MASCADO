test:
	python -m unittest discover --start-directory tests

docupdate:
	rm -rf docs/source/new_mascado.*
	cd docs; \
	sphinx-apidoc -MeT -o source/ ../new_mascado

docbuild:
	cd docs; \
	make html

docshow:
	xdg-open docs/build/html/py-modindex.html
