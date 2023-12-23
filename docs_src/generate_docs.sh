DOCS=../docs
rm -rf $DOCS
python generate_api_src.py
make clean html
cp -R build/html $DOCS
rm -rf $DOCS/reports
rm $DOCS/.buildinfo
touch $DOCS/.nojekyll