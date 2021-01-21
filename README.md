# Ruben's personal website. 

This site was built using [Jekyll](https://jekyllrb.com/), a static site generator that generates static .html pages based on the [Liquid templating system](https://github.com/Shopify/liquid/wiki/Liquid-for-Designers).

To build and run this site locally (OSX):
### Serve it up!
* Install Jekyll
    ```
    gem install --user-install bundler jekyll
    ```
* Run the following in the root of this repo. This will start a local server with all of the generated html files stored in `_site/` at <http://127.0.0.1:4000>
    ```
    bundle exec jekyll serve 
    ```

* If you don't have ruby installed, go through the below instructions to install locally (OSX)


### Install Ruby (rbenv method) from: [jekyllrb.com/docs](https://jekyllrb.com/docs/installation/macos/)
1.  Install Homebrew
    
    ```
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
2. Install rbenv and ruby-build
    
    ```
    brew install rbenv
    ```
3. Set up rbenv integration with your shell

    ```
    rbenv init
    ```
4. Restart shell and install desired Ruby version:
   ```
    rbenv install 2.7.2
    rbenv global 2.7.2
    ```
