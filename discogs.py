import csv
import json
import requests
import argparse
from time import sleep

artists = {}
releases = {}

artists_csvfile = open('artists-header.csv', 'wt', newline='')
fieldnames = ['id:ID', 'name', 'url', ':LABEL']
artists_writer = csv.DictWriter(artists_csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
artists_writer.writeheader()
artists_csvfile.close()
artists_csvfile = open('artists.csv', 'wt', newline='')
artists_writer = csv.DictWriter(artists_csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

releases_csvfile = open('releases-header.csv', 'wt', newline='')
fieldnames = ['id:ID', 'name', 'url', ':LABEL']
releases_writer = csv.DictWriter(releases_csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
releases_writer.writeheader()
releases_csvfile.close()
releases_csvfile = open('releases.csv', 'wt', newline='')
releases_writer = csv.DictWriter(releases_csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

relationships_csvfile = open('relationships-header.csv', 'wt', newline='')
fieldnames=[':START_ID', 'role', ':END_ID', ':TYPE']
relationships_writer = csv.DictWriter(relationships_csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
relationships_writer.writeheader()
relationships_csvfile.close()
relationships_csvfile = open('relationships.csv', 'wt', newline='')
relationships_writer = csv.DictWriter(relationships_csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

def follow_artist(artist_url, max_degree=None, degree=0, from_artist_id=None, from_release_id=None, reverse=False, relationship=None, role=''):
    if artist_url == '':
        return
    while True:
        sleep(2)
        r = requests.get(artist_url, headers={'User-agent': 'FooBarApp/3.0'})
        if r.status_code == 200 or r.status_code == 404:
            break
    response = json.loads(r.text)
    if 'name' not in response or r.status_code == 404: # artist not found
        return
    artist_id = 'artist/' + artist_url[artist_url.rfind('/') + 1:]
    if artist_url not in artists:
        artists_writer.writerow({'id:ID': artist_id, 'name': response['name'].replace('"', "'"), 'url': response['uri'], ':LABEL': 'ARTIST'})
        artists_csvfile.flush()
    from_id = from_artist_id if from_artist_id else from_release_id
    if from_id:
        if reverse:
            relationships_writer.writerow({':START_ID': artist_id, 'role': '', ':END_ID': from_id, ':TYPE': relationship})
        else:
            relationships_writer.writerow({':START_ID': from_id, 'role': '', ':END_ID': artist_id, ':TYPE': relationship})
    relationships_csvfile.flush()

    if artist_url in artists:
        return # I've been here before
    artists[artist_url] = response['name']
    print(f'{len(artists)}. {artists[artist_url]} {artist_url}')
    if degree >= max_degree:
        return
    for member in response.get('members', range(0, 0)):
        follow_artist(member['resource_url'], max_degree, degree + 1, from_artist_id=artist_id, relationship='PLAYED_IN', reverse=True)
    for alias in response.get('aliases', range(0, 0)):
        follow_artist(alias['resource_url'], max_degree, degree + 1, from_artist_id=artist_id, relationship='AKA')
    for group in response.get('groups', range(0, 0)):
        follow_artist(group['resource_url'], max_degree, degree + 1, from_artist_id=artist_id, relationship='PLAYED_IN')

    while True:
        sleep(2)
        r = requests.get(response['releases_url'], headers={'User-agent': 'FooBarApp/3.0'})
        if r.status_code == 200:
            break
    response_ = json.loads(r.text)
    while True:
        for release in response_['releases']:
            release_url = release['resource_url']
            
            while True:
                sleep(2)
                r = requests.get(release_url, headers={'User-agent': 'FooBarApp/3.0'})
                if r.status_code == 200:
                    break
            response__ = json.loads(r.text)
            release_id = 'release/' + release_url[release_url.rfind('/') + 1:]
            if release_url not in releases:
                releases_writer.writerow({'id:ID': release_id, 'name': release['title'].replace('"', "'"), 'url': response__['uri'], ':LABEL': 'RELEASE'})
                releases_csvfile.flush()
            relationships_writer.writerow({':START_ID': artist_id, 'role': '', ':END_ID': release_id, ':TYPE': 'APPEARS_ON'})
            relationships_csvfile.flush()

            if release_url in releases:
                continue # I've been here before
            releases[release_url] = release['title']
            for track in response__.get('tracklist', range(0, 0)):
                for artist in track.get('artists', range(0, 0)):
                    follow_artist(artist['resource_url'], max_degree, degree + 1, from_release_id=release_id, role=artist.get('role', ''), relationship='APPEARS_ON', reverse=True)
                for extraartist in track.get('extraartists', range(0, 0)):
                    follow_artist(extraartist['resource_url'], max_degree, degree + 1, from_release_id=release_id, role=extraartist.get('role', ''), relationship='APPEARS_ON', reverse=True)

        if 'next' not in response_['pagination']['urls']:
            break
        while True:
            sleep(2)
            r = requests.get(response_['pagination']['urls']['next'], headers={'User-agent': 'FooBarApp/3.0'})
            if r.status_code == 200:
                break
        response_ = json.loads(r.text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=str, help='Discogs artist ID')
    parser.add_argument('degree', type=int, help='Maximum deegres of separation')
    args = parser.parse_args()
    
    try:
        follow_artist(f'https://api.discogs.com/artists/{args.id}', max_degree=args.degree)
    except Exception as e:
        print(e)
    finally:
        artists_csvfile.close()
        releases_csvfile.close()
        relationships_csvfile.close()